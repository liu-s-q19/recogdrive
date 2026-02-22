#!/bin/bash
# set -xeuo pipefail
# export ADDR2LINE="/data/miniconda/envs/navsim/bin/x86_64-conda-linux-gnu-addr2line"

# 加载 conda 配置
source /data/miniconda/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate navsim

# 切换到代码根目录 (非常重要，否则 python 找不到模块)
cd /data/liushiqi/recogdrive || exit

# ----------------- 1. 核心路径配置 -----------------
PROJECT_ROOT="/data/liushiqi/recogdrive"
TRAIN_TEST_SPLIT=navtrain

# 环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/dataset/navsim"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ----------------- DataLoader / SHM 稳定性 -----------------
export TORCH_SHARING_STRATEGY=${TORCH_SHARING_STRATEGY:-file_system}
export TMPDIR=${TMPDIR:-$NAVSIM_EXP_ROOT/tmp}
mkdir -p "$TMPDIR"

# [输入] Stage 1 VLM 权重
VLM_PATH="$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B"

# [输入] 训练数据缓存 (Stage 2 用的)
CACHE_PATH="$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train"

# [输入] 训练集评分缓存 (你刚刚生成的 metric_cache_train)
METRIC_CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache_train"

# [输入] Stage 2 最佳模型 (Teacher/Reference)
# 约定：优先使用 EMA 权重作为 teacher/reference（通常更稳、更强），
# 若 EMA 不存在则回退到普通 last.ckpt。
DEFAULT_CKPT_DIR="$NAVSIM_EXP_ROOT/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints"
DEFAULT_CKPT_EMA="$DEFAULT_CKPT_DIR/last-EMA.ckpt"
DEFAULT_CKPT_RAW="$DEFAULT_CKPT_DIR/last.ckpt"

if [ -n "${CHECKPOINT:-}" ]; then
    :
elif [ -f "$DEFAULT_CKPT_EMA" ]; then
    CHECKPOINT="$DEFAULT_CKPT_EMA"
elif [ -f "$DEFAULT_CKPT_RAW" ]; then
    CHECKPOINT="$DEFAULT_CKPT_RAW"
else
    echo "[ERROR] No checkpoint found. Please export CHECKPOINT=/path/to/model.ckpt"
    echo "        Tried: $DEFAULT_CKPT_EMA"
    echo "        Tried: $DEFAULT_CKPT_RAW"
    exit 1
fi

# [输出] Stage 3 RL 结果目录
RL_ALGO=reinforce
OUTPUT_DIR="$NAVSIM_EXP_ROOT/recogdrive_stage3_rl_reinforce"

# ----------------- 2. 分布式配置：单机 8 卡 -----------------
GPUS_PER_NODE=8
NNODES=1

# 自动探测 Master IP
if [ -n "$PET_MASTER_ADDR" ]; then
    MASTER_ADDR=$PET_MASTER_ADDR
    MASTER_PORT=${PET_MASTER_PORT:-29500}
    NODE_RANK=0
elif [ -n "$MLP_WORKER_0_HOST" ]; then
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=${MLP_WORKER_0_PORT:-29500}
    NODE_RANK=0
else
    # 单机回退
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=29500
    NODE_RANK=0
fi

echo "=================================================="
echo "   🚀 Stage 3 RL Training (Config Aligned)"
echo "=================================================="
echo "Master: $MASTER_ADDR | Rank: $NODE_RANK | GPUs: $GPUS_PER_NODE"
echo "Batch Size: 8 (Single-node total: $((8 * GPUS_PER_NODE)))"
echo "Checkpoint: $CHECKPOINT"
echo "=================================================="

# ----------------- 3. 环境优化 -----------------
export NCCL_IB_DISABLE=1  # 依然建议禁用IB以防卡死，除非你很确定环境支持
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export CUDA_LAUNCH_BLOCKING=1

# ---------------- 4. GRPO 算法超参数 ----------------
# 数值裁剪相关
GRPO_GAMMA=0.6
GRPO_CLIP_LOW=0.00   # 裁剪百分比下界
GRPO_CLIP_HIGH=1.00  # 裁剪百分比上界
GRPO_RANDN_CLIP=5.0
GRPO_DENOISED_CLIP=1.0

# 探索与概率相关
GRPO_MIN_SAMPLING_STD=0.04
GRPO_MIN_LOGPROB_STD=0.1

# 奖励函数权重 (PDM Scorer)
SCORE_PROGRESS=10.0
SCORE_TTC=5.0
SCORE_COMFORT=2.0


# ----------------- 5. 启动命令 (严格对齐原项目参数) -----------------

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.grpo=True \
    +agent.rl_algo_type=${RL_ALGO} \
    +agent.grpo_cfg.gamma_denoising=${GRPO_GAMMA} \
    +agent.grpo_cfg.clip_advantage_lower_quantile=${GRPO_CLIP_LOW} \
    +agent.grpo_cfg.clip_advantage_upper_quantile=${GRPO_CLIP_HIGH} \
    +agent.grpo_cfg.randn_clip_value=${GRPO_RANDN_CLIP} \
    +agent.grpo_cfg.denoised_clip_value=${GRPO_DENOISED_CLIP} \
    +agent.grpo_cfg.min_sampling_denoising_std=${GRPO_MIN_SAMPLING_STD} \
    +agent.grpo_cfg.min_logprob_denoising_std=${GRPO_MIN_LOGPROB_STD} \
    \
    +agent.grpo_cfg.scorer_config.progress_weight=${SCORE_PROGRESS} \
    +agent.grpo_cfg.scorer_config.ttc_weight=${SCORE_TTC} \
    +agent.grpo_cfg.scorer_config.comfortable_weight=${SCORE_COMFORT} \
    \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path=$CHECKPOINT \
    agent.dit_type="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path=$METRIC_CACHE_PATH \
    agent.reference_policy_checkpoint=$CHECKPOINT \
    trainer.params.max_epochs=10 \
    dataloader.params.batch_size=8 \
    trainer.params.num_nodes=$NNODES \
    trainer.params.devices=$GPUS_PER_NODE \
    experiment_name=training_internvl_agent_dit_rl \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    output_dir=$OUTPUT_DIR \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    worker=sequential > train_rl_rank${NODE_RANK}.log 2>&1