from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os
import time
import gc

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    缓存特征的辅助函数，已针对多机多卡优化
    """
    if not args:
        return []
    # 1. 获取当前进程的 Rank 信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    
    # 2. 错峰加载：按照 local_rank 顺序延迟启动，避免显存瞬时峰值
    # 每台机器的 8 个进程会依次间隔 2 秒加载模型
    time.sleep(local_rank * 2)

    # 3. 强制设备隔离：确保当前进程只使用分配给它的显卡
    torch.cuda.set_device(local_rank)
    gc.collect()
    torch.cuda.empty_cache()

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # 4. 显存保护加载：强制半精度 (bfloat16)
    logger.info(f"[Rank {local_rank}] Loading agent with bfloat16 on device cuda:{local_rank}...")
    torch.set_default_dtype(torch.bfloat16)
    
    with torch.no_grad():
        agent: AbstractAgent = instantiate(cfg.agent)
        # 确保模型被推送到正确的 GPU
        if hasattr(agent, "model"):
            agent.model.to(f"cuda:{local_rank}")
            agent.model.eval()
            
    torch.set_default_dtype(torch.float32) # 恢复默认精度

    # 5. 常规 SceneLoader 配置
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )
    
    logger.info(f"Node {node_id}, Rank {local_rank}: Processing {len(scene_loader.tokens)} scenarios.")

    # 6. 执行缓存
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    
    # 显存清理
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # 基础分布式参数获取
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    
    # 初始化当前进程的设备
    torch.cuda.set_device(local_rank)

    logger.info(f"Global Rank {rank}, Local Rank {local_rank} starting...")
    pl.seed_everything(0)

    # 必须强制 worker 为 sequential，因为外部 torchrun 已经是进程级并行了
    # 这一步是为了防止进程嵌套爆内存
    logger.info("Building Sequential Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building SceneLoader for main task partition")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    # 数据分发逻辑
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    # 每个 rank 仅处理自己分片，避免多进程重复写同一缓存文件
    sharded_data_points = [dp for idx, dp in enumerate(data_points) if idx % world_size == rank]
    logger.info(
        f"Rank {rank}/{world_size} assigned {len(sharded_data_points)} logs "
        f"(total logs: {len(data_points)})."
    )

    # 执行并行映射
    _ = worker_map(worker, cache_features, sharded_data_points)
    logger.info(f"Rank {rank} finished caching task.")

if __name__ == "__main__":
    main()