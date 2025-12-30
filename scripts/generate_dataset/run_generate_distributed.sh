#!/bin/bash
# ----------------- 1. 基础环境初始化 -----------------
# 加载 conda 配置
source /home/luban/miniconda3/etc/profile.d/conda.sh
# 激活虚拟环境
conda activate navsim

# 切换到代码根目录 (NFS 共享目录)
PROJECT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive"
cd $PROJECT_ROOT
echo "Working Directory: $(pwd)"

# ----------------- 2. 自动化分布式配置 -----------------
# [A] 显卡配置
GPUS_PER_NODE=8
# 如果平台没有注入 NNODES 变量，默认尝试自动探测或设为 2
if [ -n "$PET_MASTER_ADDR" ]; then
    echo ">> Detected Luban/PET Environment"
    MASTER_ADDR=$PET_MASTER_ADDR
    MASTER_PORT=${PET_MASTER_PORT:-29500}
    NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}
    NNODES=${PET_NNODES:-1}
elif [ -n "$VC_MASTER_HOSTS" ]; then
    echo ">> Detected Volcano Environment"
    MASTER_ADDR=$(echo $VC_MASTER_HOSTS | cut -d',' -f1)
    MASTER_PORT=${MASTER_PORT:-29500}
    NODE_RANK=${VC_TASK_INDEX:-0}
    NNODES=$(echo $VC_MASTER_HOSTS | tr ',' '\n' | wc -l)
elif [ -n "$MLP_WORKER_0_HOST" ]; then
    echo ">> Detected Standard MLP Environment"
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=${MLP_WORKER_0_PORT:-29500}
    NODE_RANK=$MLP_ROLE_INDEX
    NNODES=${MLP_WORKER_NUM:-1}
else
    echo ">> Warning: No distributed variables found. Fallback to Localhost (Single Node)."
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=29500
    NODE_RANK=0
    NNODES=1
fi

# 你的集群是2机8卡，如果自动探测不准，这里取消注释强制指定
# NNODES=2 

echo "=================================================="
echo "   🚀 Cluster Generation Job Start"
echo "=================================================="
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"
echo "My Rank:     $NODE_RANK"
echo "Total Nodes: $NNODES"
echo "GPUs/Node:   $GPUS_PER_NODE"
echo "Total GPUs:  $((NNODES * GPUS_PER_NODE))"
echo "=================================================="

# ----------------- 3. 环境变量导出 -----------------
# 这里的 navtrain 只是为了让 Hydra 能找到一个基础配置文件初始化
# 实际上我们的 Python 代码里已经重写了 filter 逻辑，所以这里用 navtrain 没问题
TRAIN_TEST_SPLIT=navtrain 

# 模型路径
MODEL_PATH="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/ckpt/ReCogDrive-VLM-8B" 

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/data/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/data/navsim"
export NAVSIM_DATA_ROOT=$OPENSCENE_DATA_ROOT
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 网络通信优化 (防止多机卡死，强制 TCP)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_FAMILY=AF_INET

# ----------------- 4. 启动分布式生成 -----------------
LOG_FILE="generation_rank${NODE_RANK}.log"

echo "Starting torchrun on Node ${NODE_RANK}..."

# 【关键修正点】：这里使用 +model_path 而不是 model_path
# 因为 model_path 很可能不在原始的 navtrain.yaml 里，用 + 表示新增参数
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE \
    scripts/generate_dataset/generate_reasoning_gt_distributed.py \
    --config-name $TRAIN_TEST_SPLIT \
    +model_path=$MODEL_PATH \
    > $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] Torchrun failed on Node ${NODE_RANK}! Check $LOG_FILE."
    exit 1
fi

echo "Node ${NODE_RANK} finished generation."

# ----------------- 5. 结果合并 (仅在 Master 节点执行) -----------------
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=================================================="
    echo "Waiting for all nodes to sync..."
    # 稍微多睡一会，防止 NFS 文件系统有延迟，导致 Rank 0 看不到其他节点生成的文件
    sleep 30 
    
    echo ">> Master Node (Rank 0): Start Merging..."
    
    python -c "
import json
import glob
import os

# 这里的名字可以根据你想要的最终文件名修改
output_name = 'reasoning_gt_trainval_full.json'
files = glob.glob('reasoning_gt_part_*.json')

print(f'Found {len(files)} part files.')
data = {}
for f in files:
    try:
        part_data = json.load(open(f))
        data.update(part_data)
        print(f'Merged {f}: {len(part_data)} samples')
    except Exception as e:
        print(f'[ERROR] merging {f}: {e}')

with open(output_name, 'w') as f:
    json.dump(data, f, indent=4) # 加了 indent=4 让最终结果也漂亮

print(f'[SUCCESS] Final merged file: {output_name} ({len(data)} total samples)')
" >> $LOG_FILE 2>&1
    
    echo "Done! Check $LOG_FILE for details."
fi