import json
import pickle
from pathlib import Path

import pytest

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.metric_caching.metric_cache import MapParameters


class DummyV2MetricCache:
    def __init__(self) -> None:
        self.scene_type = "original"
        self.human_trajectory = object()
        self.past_human_trajectory = object()
        self.map_parameters = MapParameters(
            map_root="/maps",
            map_version="nuplan-maps-v1.0",
            map_name="test-map",
        )


class DummyLegacyMetricCache:
    def __init__(self) -> None:
        self.scene_type = "original"


def _write_cache_index(cache_root: Path, token: str, payload: object) -> Path:
    token_dir = cache_root / "log" / "unknown" / token
    token_dir.mkdir(parents=True)
    cache_file = token_dir / "metric_cache.pkl"
    with cache_file.open("wb") as file_obj:
        pickle.dump(payload, file_obj)

    metadata_dir = cache_root / "metadata"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "index.csv"
    metadata_file.write_text(f"file_path\n{cache_file}\n", encoding="utf-8")
    return cache_file


def test_metric_cache_loader_rejects_legacy_cache_without_v2_marker(tmp_path: Path) -> None:
    token_dir = tmp_path / "log" / "scenario" / "token"
    token_dir.mkdir(parents=True)
    (token_dir / "metric_cache.pkl").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="cache_meta.json"):
        MetricCacheLoader(tmp_path, require_v2_metadata=True)


def test_metric_cache_loader_accepts_v2_marker(tmp_path: Path) -> None:
    token_dir = tmp_path / "log" / "scenario" / "token"
    token_dir.mkdir(parents=True)
    (token_dir / "metric_cache.pkl").write_bytes(b"legacy")
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "navsim_v2_recogdrive_1",
                "train_test_split": "navhard_two_stage",
                "scene_loader_mode": "navsim_v2_scene_loader",
            }
        )
    )

    loader = MetricCacheLoader(tmp_path, require_v2_metadata=True)

    assert loader.tokens == ["token"]


def test_prepare_metric_cache_metadata_adopts_external_cache(tmp_path: Path) -> None:
    from navsim.planning.script.run_prepare_metric_cache_metadata import prepare_metric_cache_metadata

    _write_cache_index(tmp_path, "token-a", DummyV2MetricCache())

    summary = prepare_metric_cache_metadata(
        cache_root=tmp_path,
        train_test_split="navhard_two_stage",
        scene_loader_mode="navsim_v2_scene_loader",
    )

    metadata = json.loads((tmp_path / "cache_meta.json").read_text(encoding="utf-8"))
    loader = MetricCacheLoader(tmp_path, require_v2_metadata=True)

    assert metadata == {
        "schema_version": "navsim_v2_recogdrive_1",
        "train_test_split": "navhard_two_stage",
        "scene_loader_mode": "navsim_v2_scene_loader",
    }
    assert summary["sample_token"] == "token-a"
    assert summary["indexed_row_count"] == 1
    assert loader.tokens == ["token-a"]


def test_prepare_metric_cache_metadata_rejects_missing_metadata_csv(tmp_path: Path) -> None:
    from navsim.planning.script.run_prepare_metric_cache_metadata import prepare_metric_cache_metadata

    with pytest.raises(ValueError, match="metadata/.+csv"):
        prepare_metric_cache_metadata(
            cache_root=tmp_path,
            train_test_split="navhard_two_stage",
            scene_loader_mode="navsim_v2_scene_loader",
        )


def test_prepare_metric_cache_metadata_rejects_legacy_metric_cache_objects(tmp_path: Path) -> None:
    from navsim.planning.script.run_prepare_metric_cache_metadata import prepare_metric_cache_metadata

    _write_cache_index(tmp_path, "token-legacy", DummyLegacyMetricCache())

    with pytest.raises(ValueError, match="missing v2 fields"):
        prepare_metric_cache_metadata(
            cache_root=tmp_path,
            train_test_split="navhard_two_stage",
            scene_loader_mode="navsim_v2_scene_loader",
        )

    assert not (tmp_path / "cache_meta.json").exists()


def test_prepare_metric_cache_metadata_rejects_conflicting_existing_metadata(tmp_path: Path) -> None:
    from navsim.planning.script.run_prepare_metric_cache_metadata import prepare_metric_cache_metadata

    _write_cache_index(tmp_path, "token-a", DummyV2MetricCache())
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "navsim_v2_recogdrive_1",
                "train_test_split": "navtest",
                "scene_loader_mode": "navsim_v2_scene_loader",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing cache_meta.json"):
        prepare_metric_cache_metadata(
            cache_root=tmp_path,
            train_test_split="navhard_two_stage",
            scene_loader_mode="navsim_v2_scene_loader",
        )
