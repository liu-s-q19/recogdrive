from __future__ import annotations

import json
import logging
import os
import pickle
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.cache_metadata import resolve_train_test_split_name
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _optional_path(value: str):
    return Path(value) if value else None


def _maybe_init_dist() -> tuple[int, int, int]:
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, rank
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    return int(os.getenv("LOCAL_RANK", "0")), 1, 0


def _build_scene_loader(cfg: DictConfig, scene_filter: SceneFilter, sensor_config: SensorConfig) -> SceneLoader:
    return SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        original_sensor_path=_optional_path(cfg.original_sensor_path),
        synthetic_sensor_path=_optional_path(cfg.get("synthetic_sensor_path")),
        synthetic_scenes_path=_optional_path(cfg.get("synthetic_scenes_path")),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def _split_tokens(tokens: List[str], rank: int, world_size: int) -> List[str]:
    return [token for index, token in enumerate(tokens) if index % world_size == rank]


def _resolve_stage_two_tokens(base_loader: SceneLoader, metric_cache_tokens: List[str]) -> List[str]:
    stage_two_tokens_source = (
        base_loader.reactive_tokens_stage_two
        if base_loader.reactive_tokens_stage_two is not None
        else list(base_loader.synthetic_scenes.keys())
    )
    return sorted(set(stage_two_tokens_source) & set(metric_cache_tokens))


def _configure_stage_filters(
    base_scene_filter: SceneFilter,
    stage_one_tokens: List[str],
    stage_two_tokens: List[str],
) -> tuple[SceneFilter, SceneFilter]:
    stage_one_filter: SceneFilter = instantiate(base_scene_filter)
    stage_one_filter.tokens = stage_one_tokens

    stage_two_filter: SceneFilter = instantiate(base_scene_filter)
    stage_two_filter.tokens = None
    stage_two_filter.synthetic_scene_tokens = stage_two_tokens

    return stage_one_filter, stage_two_filter


def _save_rank_results(output_dir: Path, rank: int, local_results: List[Dict[str, Any]]) -> Path:
    partial_dir = output_dir / "partial_rank_results"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"rank_{rank:02d}_results.pkl"
    with partial_path.open("wb") as file_obj:
        pickle.dump(local_results, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return partial_path


def _merge_rank_results(output_dir: Path, world_size: int) -> List[Dict[str, Any]]:
    partial_dir = output_dir / "partial_rank_results"
    merged_results: List[Dict[str, Any]] = []
    for rank in range(world_size):
        partial_path = partial_dir / f"rank_{rank:02d}_results.pkl"
        with partial_path.open("rb") as file_obj:
            merged_results.extend(pickle.load(file_obj))
    return merged_results


def _evaluate_tokens(
    cfg: DictConfig,
    agent: AbstractAgent,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_loader: MetricCacheLoader,
    scene_filter: SceneFilter,
    tokens: List[str],
    stage_name: str,
) -> List[Dict[str, Any]]:
    scene_loader = _build_scene_loader(cfg, scene_filter, agent.get_sensor_config())
    train_data = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=True,
        is_decoder=True,
        cache_loader_mode=cfg.get("cache_loader_mode", "navsim_v2_scene_loader"),
        runtime_cache_version=cfg.get("runtime_cache_version"),
        train_test_split_name=resolve_train_test_split_name(cfg),
    )

    results: List[Dict[str, Any]] = []
    for token in tokens:
        score_row: Dict[str, Any] = {"token": token, "valid": True, "stage": stage_name}
        try:
            metric_cache: MetricCache = metric_cache_loader.get_from_token(token)
            train_data.load_token_cache(token)
            features, _targets = train_data._load_scene_with_token(token)
            trajectory = agent.compute_trajectory(features)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception:
            logger.warning("Agent failed for token %s in %s", token, stage_name)
            traceback.print_exc()
            score_row["valid"] = False
        results.append(score_row)
    return results


def _build_summary(score_df: pd.DataFrame) -> Dict[str, float]:
    valid_df = score_df[score_df["valid"]]
    stage_one_df = valid_df[valid_df["stage"] == "stage_one"]
    stage_two_df = valid_df[valid_df["stage"] == "stage_two"]

    summary = {
        "extended_pdm_score_stage_one": float(stage_one_df["score"].mean()) if not stage_one_df.empty else float("nan"),
        "extended_pdm_score_stage_two": float(stage_two_df["score"].mean()) if not stage_two_df.empty else float("nan"),
        "extended_pdm_score_combined": float(valid_df["score"].mean()) if not valid_df.empty else float("nan"),
    }
    summary["final_extended_pdm_score"] = summary["extended_pdm_score_combined"]
    return summary


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank, world_size, rank = _maybe_init_dist()
    build_logger(cfg)

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    base_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    base_loader = _build_scene_loader(cfg, base_scene_filter, SensorConfig.build_no_sensors())
    metric_cache_loader = MetricCacheLoader(
        Path(cfg.metric_cache_path),
        require_v2_metadata=True,
        expected_schema_version=cfg.get("runtime_cache_version"),
    )

    stage_one_tokens = sorted(set(base_loader.tokens_stage_one) & set(metric_cache_loader.tokens))
    stage_two_tokens = _resolve_stage_two_tokens(base_loader, metric_cache_loader.tokens)

    local_stage_one_tokens = _split_tokens(stage_one_tokens, rank, world_size)
    local_stage_two_tokens = _split_tokens(stage_two_tokens, rank, world_size)

    stage_one_filter, stage_two_filter = _configure_stage_filters(
        cfg.train_test_split.scene_filter,
        stage_one_tokens=local_stage_one_tokens,
        stage_two_tokens=local_stage_two_tokens,
    )

    local_results: List[Dict[str, Any]] = []
    local_results.extend(
        _evaluate_tokens(
            cfg,
            agent,
            simulator,
            scorer,
            metric_cache_loader,
            stage_one_filter,
            local_stage_one_tokens,
            "stage_one",
        )
    )
    local_results.extend(
        _evaluate_tokens(
            cfg,
            agent,
            simulator,
            scorer,
            metric_cache_loader,
            stage_two_filter,
            local_stage_two_tokens,
            "stage_two",
        )
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_rank_results(output_dir, rank, local_results)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        final_results = _merge_rank_results(output_dir, world_size)
        score_df = pd.DataFrame(final_results)
        summary = _build_summary(score_df)

        average_row = score_df.drop(columns=["token", "valid", "stage"], errors="ignore").mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = score_df["valid"].all()
        average_row["stage"] = "combined"
        for key, value in summary.items():
            average_row[key] = value
        score_df.loc[len(score_df)] = average_row

        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        csv_path = output_dir / f"{timestamp}.csv"
        score_df.to_csv(csv_path, index=False)
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, sort_keys=True)
        logger.info("Saved navhard two-stage results to %s and %s", csv_path, summary_path)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
