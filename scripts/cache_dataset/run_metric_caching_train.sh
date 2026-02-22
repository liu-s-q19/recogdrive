#!/bin/bash

# ================= 1. 环境加载 =================
source /data/miniconda/etc/profile.d/conda.sh
conda activate navsim
cd /data/liushiqi/recogdrive || exit

#!/bin/bash
set -x

# ----------------- 1. 路径与环境配置 -----------------
# 【修改点1】目标改为训练集
TRAIN_TEST_SPLIT=navtrain

# 你的项目根目录 (保持不变)
PROJECT_ROOT="/data/liushiqi/recogdrive"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/dataset/navsim"

# 【修改点2】输出路径加个后缀 _train，避免覆盖评估集的缓存
CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_train"

# ----------------- 2. 启动命令 (极速版配置) -----------------

# 【修改点3 & 4】修改实验名和日志文件名
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH \
    worker=single_machine_thread_pool \
    worker.max_workers=64 \
    worker.use_process_pool=True \
    +experiment_name=metric_caching_navtrain > metric_caching_train.log 2>&1