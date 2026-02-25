from typing import Tuple
from pathlib import Path
import logging
import os
import datetime
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict
from callback import EMA, EMAModelCheckpoint
import pytorch_lightning.callbacks as plc

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _configure_rank_local_tmpdirs(rank: int, local_rank: int) -> None:
    """Isolate tmp/compiler caches per process to avoid multi-process races.

    Each torchrun local worker previously inherited the same TMPDIR from launcher,
    which could cause races in multiprocessing finalizers and inductor/triton caches.
    """
    base_tmp = os.getenv("RUNTIME_TMP_BASE") or "/dev/shm/rdtmp"
    node_rank = os.getenv("NODE_RANK", "0")

    process_tmp = os.path.join(base_tmp, f"n{node_rank}", f"r{rank}l{local_rank}")
    os.makedirs(process_tmp, exist_ok=True)

    os.environ["TMPDIR"] = process_tmp
    os.environ["TMP"] = process_tmp
    os.environ["TEMP"] = process_tmp

    inductor_cache = os.path.join(process_tmp, "torchinductor_cache")
    triton_cache = os.path.join(process_tmp, "triton_cache")
    os.makedirs(inductor_cache, exist_ok=True)
    os.makedirs(triton_cache, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
    os.environ["TRITON_CACHE_DIR"] = triton_cache

    logger.info(
        "Per-rank tmp configured: TMPDIR=%s, TORCHINDUCTOR_CACHE_DIR=%s, TRITON_CACHE_DIR=%s",
        process_tmp,
        inductor_cache,
        triton_cache,
    )


def _maybe_disable_torch_compile() -> None:
    """Optionally disable torch.compile/dynamo for stability on large distributed runs."""
    if os.getenv("DISABLE_TORCH_COMPILE", "0") != "1":
        return
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    logger.warning("DISABLE_TORCH_COMPILE=1 -> TORCHDYNAMO_DISABLE=1 (torch.compile disabled)")


def _get_dist_timeout() -> datetime.timedelta:
    """Return process-group timeout.

    In multi-node runs, some ranks can be much slower (e.g., cache scanning on network FS).
    A larger timeout avoids spurious monitoredBarrier failures.
    """
    minutes_str = os.getenv("DIST_TIMEOUT_MINUTES") or os.getenv("DIST_TIMEOUT_MIN") or "60"
    try:
        minutes = int(minutes_str)
    except ValueError:
        minutes = 60
    return datetime.timedelta(minutes=minutes)


def _maybe_set_torch_sharing_strategy() -> None:
    """Optionally switch PyTorch CPU tensor sharing strategy.

    On some clusters/containers, /dev/shm is tiny, causing DataLoader worker IPC to fail with:
    'unable to write to file </torch_...>: No space left on device' and bus errors.

    Set env var TORCH_SHARING_STRATEGY=file_system to route IPC via temp files (respects TMPDIR).
    """
    strategy = os.getenv("TORCH_SHARING_STRATEGY")
    if not strategy:
        return
    try:
        torch.multiprocessing.set_sharing_strategy(strategy)
        logger.info("Set torch sharing strategy: %s", strategy)
    except Exception as e:
        logger.warning("Failed to set torch sharing strategy '%s': %s", strategy, e)


def load_callbacks():
    callbacks = []

    use_ema = True
    if use_ema:
        callbacks.append(
            EMAModelCheckpoint(
                monitor='val/loss_epoch',
                mode='min',
                save_last=True,
                save_on_train_epoch_end=True,
                save_top_k=5,
                every_n_epochs=1,
            )
        )
        callbacks.append(EMA(decay=0.999))
    else:
        callbacks.append(
            plc.ModelCheckpoint(
                monitor='val/loss_epoch',
                mode='min',
                save_last=True,
                save_on_train_epoch_end=True,
                save_top_k=5,
                every_n_epochs=1,
            )
        )

    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
    return callbacks


def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).detach().cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).detach().cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).detach().cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).detach().cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).detach().cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'last_hidden_state': last_hidden_state,
        'status_feature': status_feature
    }

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    _configure_rank_local_tmpdirs(rank=rank, local_rank=local_rank)
    _maybe_disable_torch_compile()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    timeout = _get_dist_timeout()

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    _maybe_set_torch_sharing_strategy()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=load_callbacks())

    logger.info("Starting Training")
    try:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
