#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"

source /data/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}"

set -x

# ----------------- 1. 基础环境与路径配置 -----------------
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"  # 评估集 Split

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${RUNTIME_ROOT}/exp}"
export NAVSIM_OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs}"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/dataset/navsim}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "${TMPDIR}"

# ----------------- 2. 显卡配置 (单机8卡) -----------------
# 评估通常不需要多机，单机 8 张 足够
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# 这里假设你使用一台机器上的 8 张卡进行并行评估
GPUS="${GPUS:-8}"
echo "GPUS: ${GPUS}"
# export CUDA_LAUNCH_BLOCKING=1

# ----------------- 3. 关键模型路径 (请务必检查!) -----------------

# [A] 你的训练产物 (Checkpoint)
# 这是你刚刚训练完的模型。通常在 checkpoints 文件夹里会有 'last.ckpt' 或者 'epoch=xx.ckpt'
# 如果你找不到这个文件，请去文件夹里确认一下具体名字！
CHECKPOINT="${CHECKPOINT:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_ema_multinode_4nodes_8gpus/lightning_logs/version_1/checkpoints/epoch=99-step=8400-EMA.ckpt}"

# [B] VLM 权重 (第一阶段产物，保持不变)
VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"

# [C] 评估集缓存路径 (来自 run_caching_..._eval.sh)
# 【注意】这个路径必须存在！如果你之前没跑通 eval 缓存脚本，这里会报错。
CACHE_PATH_EVAL="${CACHE_PATH_EVAL:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_test}"

# ----------------- 自动提取 EXP_NAME -----------------
# 1. ${CHECKPOINT#*outputs/} : 从左边开始删除，直到找到 'outputs/' 为止
#    结果变成: rl_A_clamp5/lightning_logs/...
temp_str="${CHECKPOINT#*outputs/}"

# 2. ${temp_str%%/*} : 从右边开始删除，保留第一个 '/' 左边的内容
#    结果变成: rl_A_clamp5
EXP_NAME="${EXP_NAME_OVERRIDE:-${temp_str%%/*}}"

echo "Detected Experiment Name: ${EXP_NAME}"
# ----------------------------------------------------

# 现在可以使用提取出来的 EXP_NAME 了
OUTPUT_DIR="${OUTPUT_DIR:-$NAVSIM_OUTPUT_ROOT/${EXP_NAME}/eval}"
ENTRYPOINT="${ENTRYPOINT:-$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recogdrive_agent_eval_shiqi}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/eval_shiqi.log}"
CACHE_LOADER_MODE="${CACHE_LOADER_MODE:-legacy_cached_features}"

mkdir -p "$OUTPUT_DIR"

# ----------------- 4. 启动评估命令 -----------------
# 默认入口是 legacy gather 版：
#   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py
# 如需切到 patched fs-merge 兜底实现，把脚本入口替换成：
#   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive_fsmerge.py
# patched 版本会先保存各 rank 的 partial 结果，再由 rank0 做文件系统合并。
torchrun \
    --nproc_per_node=${GPUS} \
    "$ENTRYPOINT" \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.cache_loader_mode="${CACHE_LOADER_MODE}" \
    cache_path=$CACHE_PATH_EVAL \
    cache_loader_mode="${CACHE_LOADER_MODE}" \
    use_cache_without_dataset=True \
    agent.sampling_method="ddim" \
    worker=sequential \
    output_dir=$OUTPUT_DIR \
    experiment_name=$EXPERIMENT_NAME > "$LOG_FILE" 2>&1
