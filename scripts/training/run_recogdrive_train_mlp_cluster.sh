#!/bin/bash
# 加载 conda 配置
source /home/luban/miniconda3/etc/profile.d/conda.sh
# 激活你的虚拟环境
conda activate navsim
# 切换到代码根目录 (非常重要，否则 python 找不到模块)
cd /nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive


# 打印所有环境变量到日志，帮我们找 IP 变量名
printenv > env_debug.log
echo "Environment variables saved to env_debug.log"
#!/bin/bash

# ----------------- 1. 路径与基础配置 -----------------
# 你的 NFS 代码根目录
PROJECT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive"
TRAIN_TEST_SPLIT=navtrain

# 导出环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/data/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/data/navsim"

# 模型与缓存路径
VLM_PATH="$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B"
CACHE_PATH="$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train"
OUTPUT_DIR="$NAVSIM_EXP_ROOT/recogdrive_stage2_training_ema_multinode_16gpus"

# ----------------- 2. 自动化分布式配置 (深度适配 Luban/PET) -----------------

# [A] 显卡配置：适配你的 8卡 H20
GPUS_PER_NODE=8
NNODES=2  # 你申请了 2 台机器

# [B] 自动获取 Master IP 和 Rank
# 根据你的 debug.log，平台使用的是 PET_ 或 DISTRIBUTED_ 前缀

if [ -n "$PET_MASTER_ADDR" ]; then
    # 适配 Luban/PET 环境
    echo "Detected Luban/PET Environment"
    MASTER_ADDR=$PET_MASTER_ADDR
    MASTER_PORT=${PET_MASTER_PORT:-29500}
    NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}
    
elif [ -n "$VC_MASTER_HOSTS" ]; then
    # 适配 Volcano 调度环境 (备选)
    echo "Detected Volcano Environment"
    # 取逗号前的第一个 host
    MASTER_ADDR=$(echo $VC_MASTER_HOSTS | cut -d',' -f1)
    MASTER_PORT=${MASTER_PORT:-29500}
    NODE_RANK=${VC_TASK_INDEX:-0}

elif [ -n "$MLP_WORKER_0_HOST" ]; then
    # 适配标准 MLP 环境 (旧版)
    echo "Detected Standard MLP Environment"
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=${MLP_WORKER_0_PORT:-29500}
    NODE_RANK=$MLP_ROLE_INDEX

else
    # 都没有，回退到单机调试模式
    echo "Warning: No distributed variables found. Fallback to localhost."
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=29500
    NODE_RANK=0
    NNODES=1
fi

echo "=================================================="
echo "   🚀 Cluster Distributed Training Start"
echo "=================================================="
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"
echo "My Rank:     $NODE_RANK"
echo "GPUs/Node:   $GPUS_PER_NODE"
echo "Total Nodes: $NNODES"
echo "=================================================="

# ----------------- 3. 环境优化 -----------------
# 强制使用 TCP 通信，避免 InfiniBand 配置问题
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_FAMILY=AF_INET

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# ... (前面的配置部分保持不变) ...

# ----------------- 4. 启动命令 -----------------
# 你的任务只允许提交一个脚本，这个命令会在所有节点上并行执行

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_ema.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=100 \
    experiment_name=training_recogdrive_agent_cluster \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    output_dir=$OUTPUT_DIR \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    worker=sequential \
    dataloader.params.batch_size=32 \
    trainer.params.num_nodes=$NNODES \
    trainer.params.devices=$GPUS_PER_NODE \
    > train_mlp_rank${NODE_RANK}.log 2>&1