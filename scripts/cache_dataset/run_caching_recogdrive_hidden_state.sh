#!/bin/bash

# ================= 1. 环境加载 =================
source /data/miniconda/etc/profile.d/conda.sh
conda activate navsim
cd /data/liushiqi/recogdrive || exit

# ================= 2. 路径配置 (NFS) =================
TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/liushiqi/recogdrive/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="/data/liushiqi/recogdrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/liushiqi/recogdrive"
export OPENSCENE_DATA_ROOT="/data/liushiqi/recogdrive/dataset/navsim"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train

export PYTHONPATH="$(pwd):${PYTHONPATH}"


# ================= 3. 显卡配置 =================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=63669

# # 关键：由于你要跑 8B 模型，建议设置这个环境变量减少显存碎片
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Launching 8-GPU Caching Task using run_dataset_caching.py..."

# ================= 4. 启动任务 =================
# 关键改动：
# 1. 指向你提供的 run_dataset_caching.py
# 2. 增加 worker=sequential，防止每个进程内部再开多线程导致 OOM
# 3. 如果还是 OOM，请将 --nproc_per_node 改为 4 (每张卡跑一个进程)

torchrun \
    --standalone \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/data/liushiqi/recogdrive/ckpt/ReCogDrive-VLM-8B" \
    cache_path=$CACHE_PATH \
    force_cache_computation=True \
    worker=sequential > caching_dataset_8gpu.txt 2>&1