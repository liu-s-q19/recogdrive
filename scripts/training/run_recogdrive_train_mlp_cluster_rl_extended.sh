#!/bin/bash
# 加载 conda 配置
source /home/luban/miniconda3/etc/profile.d/conda.sh
# 激活你的虚拟环境
conda activate navsim
# 切换到代码根目录 (非常重要，否则 python 找不到模块)
cd /nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive

# ----------------- 1. 基础路径 -----------------
PROJECT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive"
TRAIN_TEST_SPLIT=navtrain

# 环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/data/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/data/navsim"

# VLM 权重
VLM_PATH="$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B"
# 数据缓存
CACHE_PATH="$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train"
METRIC_CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_train"

# ==============================================================================
# 👇 【核心修改区】 👇
# ==============================================================================

# [1. 初始权重]：指向你【刚刚跑完】的 Stage 3 结果
# 这是为了“接力”继续跑。请检查路径是否正确！
INIT_CHECKPOINT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/exp/recogdrive_stage3_rl_training_16gpus_bs8/lightning_logs/version_0/checkpoints/epoch=9-step=6650.ckpt"

# [2. 参考权重]：依然指向【Stage 2】的最佳 EMA 模型
# 这是为了“不忘初心”，防止模型为了刷分而动作变形。
REF_CHECKPOINT="$NAVSIM_EXP_ROOT/recogdrive_stage2_training_ema_multinode_16gpus/lightning_logs/version_0/checkpoints/epoch=95-step=16032-EMA.ckpt"

# [3. 输出目录]：改个新名字，别覆盖旧的
OUTPUT_DIR="$NAVSIM_EXP_ROOT/recogdrive_stage3_rl_training_16gpus_bs8_extended"

# [4. 实验名称]：Hydra 记录用的名字
EXP_NAME="training_recogdrive_rl_extended"

# ==============================================================================

# ----------------- 2. 自动化分布式配置 (保持不变) -----------------
GPUS_PER_NODE=8
NNODES=2

if [ -n "$PET_MASTER_ADDR" ]; then
    MASTER_ADDR=$PET_MASTER_ADDR
    MASTER_PORT=${PET_MASTER_PORT:-29500}
    NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}
elif [ -n "$MLP_WORKER_0_HOST" ]; then
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=${MLP_WORKER_0_PORT:-29500}
    NODE_RANK=$MLP_ROLE_INDEX
else
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=29500
    NODE_RANK=0
    NNODES=1
fi

echo "=================================================="
echo "   🚀 Stage 3 Extended Training (Round 2)"
echo "=================================================="
echo "Init Model (Student): $INIT_CHECKPOINT"
echo "Ref Model (Teacher):  $REF_CHECKPOINT"
echo "Output Dir:           $OUTPUT_DIR"
echo "=================================================="

# ----------------- 3. 环境优化 -----------------
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# ----------------- 4. 启动命令 -----------------

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-5 \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path=$METRIC_CACHE_PATH \
    \
    agent.checkpoint_path="'$INIT_CHECKPOINT'" \
    agent.reference_policy_checkpoint="'$REF_CHECKPOINT'" \
    \
    trainer.params.max_epochs=10 \
    dataloader.params.batch_size=8 \
    trainer.params.num_nodes=$NNODES \
    trainer.params.devices=$GPUS_PER_NODE \
    experiment_name=$EXP_NAME \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    output_dir=$OUTPUT_DIR \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    worker=sequential > train_rl_extended_rank${NODE_RANK}.log 2>&1