#!/bin/bash

# ================= 1. 环境加载 =================
source /data/miniconda/etc/profile.d/conda.sh
conda activate navsim
cd /data/liushiqi/recogdrive || exit

# ================= 2. 路径与变量配置 =================
TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/liushiqi/recogdrive/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="/data/liushiqi/recogdrive/exp"
export NAVSIM_DEVKIT_ROOT="/data/liushiqi/recogdrive"
export OPENSCENE_DATA_ROOT="/data/liushiqi/recogdrive/dataset/navsim"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_test
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ================= 3. 分布式多机配置 =================
MASTER_ADDR="10.199.7.32"
MASTER_PORT=63669

# 从命令行参数获取 RANK，如果没有传则默认为 0
NODE_RANK=${1:-0} 

echo "🚀 Launching Multi-Node (4 Nodes, 32 GPUs) Task..."
echo "Node Rank: $NODE_RANK | Master: $MASTER_ADDR:$MASTER_PORT"

# ================= 4. 启动任务 =================
# 去掉了 --standalone
# --nnodes=4 表示总共4台机器
# --node_rank=$NODE_RANK 告诉脚本当前是第几台
# --rdzv_endpoint 指定主节点地址

torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --rdzv_id=navsim_dist_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/data/liushiqi/recogdrive/ckpt/ReCogDrive-VLM-8B" \
    cache_path=$CACHE_PATH \
    worker=sequential > caching_dataset_test_node_${NODE_RANK}.txt 2>&1