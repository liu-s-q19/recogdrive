#!/bin/bash
set -x

# ----------------- 1. 路径与环境配置 -----------------
TRAIN_TEST_SPLIT=navtest
PROJECT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/data/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/data/navsim"

CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache"

# ----------------- 2. 启动命令 (多线程加速版) -----------------
# 修改点 A: worker.num_workers -> worker.max_workers (修正参数名)
# 修改点 B: 加上 +experiment_name (修复之前的报错)

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH \
    worker=single_machine_thread_pool \
    worker.max_workers=64 \
    worker.use_process_pool=True \
    +experiment_name=metric_caching_navtest > metric_caching.log 2>&1