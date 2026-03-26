#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"

source /data/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}" || exit 1

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${RUNTIME_ROOT}/exp}"
export NAVSIM_OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/dataset/navsim}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_descriptor}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
mkdir -p "${TMPDIR}"

VLM_PATH="${VLM_PATH:-${PROJECT_ROOT}/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH="${CACHE_PATH:-/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_train}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${CHECKPOINT:-/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints}}"

if [[ -d "${INIT_CHECKPOINT}" ]]; then
  DEFAULT_CKPT_DIR="${INIT_CHECKPOINT}"
else
  DEFAULT_CKPT_DIR="$(dirname "${INIT_CHECKPOINT}")"
fi

DEFAULT_CKPT_EMA="${DEFAULT_CKPT_DIR}/last-EMA.ckpt"
DEFAULT_CKPT_RAW="${DEFAULT_CKPT_DIR}/last.ckpt"

if [[ -f "${INIT_CHECKPOINT}" ]]; then
  :
elif [[ -f "${DEFAULT_CKPT_EMA}" ]]; then
  INIT_CHECKPOINT="${DEFAULT_CKPT_EMA}"
elif [[ -f "${DEFAULT_CKPT_RAW}" ]]; then
  INIT_CHECKPOINT="${DEFAULT_CKPT_RAW}"
else
  echo "[ERROR] No checkpoint file found."
  echo "        Input INIT_CHECKPOINT: ${INIT_CHECKPOINT}"
  echo "        Tried: ${DEFAULT_CKPT_EMA}"
  echo "        Tried: ${DEFAULT_CKPT_RAW}"
  exit 1
fi

REFERENCE_POLICY_CHECKPOINT="${REFERENCE_POLICY_CHECKPOINT:-${INIT_CHECKPOINT}}"

OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_OUTPUT_ROOT}/grpo/rpp1n8g_g16_bcfloor05_refkl002_${TS}}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_rank0_${TS}.log}"
CACHE_LOADER_MODE="${CACHE_LOADER_MODE:-legacy_cached_features}"
USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-true}"
REAL_EVAL_ENABLED="${REAL_EVAL_ENABLED:-1}"
REAL_EVAL_SPLIT="${REAL_EVAL_SPLIT:-navhard_two_stage}"
REAL_EVAL_ASYNC_MODE="${REAL_EVAL_ASYNC_MODE:-tmux}"
REAL_EVAL_POLL_INTERVAL_SEC="${REAL_EVAL_POLL_INTERVAL_SEC:-120}"
REAL_EVAL_TOP_K="${REAL_EVAL_TOP_K:-3}"
REAL_EVAL_KEEP_LAST="${REAL_EVAL_KEEP_LAST:-1}"
REAL_EVAL_SCORE_DECIMALS="${REAL_EVAL_SCORE_DECIMALS:-6}"
REAL_EVAL_GPUS="${REAL_EVAL_GPUS:-8}"
REAL_EVAL_SESSION_PREFIX="${REAL_EVAL_SESSION_PREFIX:-eval-navhard-rpp}"
REAL_EVAL_WATCHER_SCRIPT="${REAL_EVAL_WATCHER_SCRIPT:-${PROJECT_ROOT}/scripts/evaluation/watch_epoch9_and_eval.sh}"
NAVHARD_METRIC_CACHE_PATH="${NAVHARD_METRIC_CACHE_PATH:-/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29741}"

RL_ALGO="${RL_ALGO:-reinforce_plus_plus}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
CKPT_MONITOR="${CKPT_MONITOR:-null}"
CKPT_MODE="${CKPT_MODE:-max}"
CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:--1}"
CKPT_EVERY_N_EPOCHS="${CKPT_EVERY_N_EPOCHS:-1}"
CKPT_SAVE_LAST="${CKPT_SAVE_LAST:-true}"
CKPT_FILENAME="${CKPT_FILENAME:-}"
if [[ -z "${CKPT_FILENAME}" ]]; then
  CKPT_FILENAME='epoch={epoch:02d}-step={step}'
fi
DATALOADER_BATCH_SIZE="${DATALOADER_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

GRPO_GAMMA="${GRPO_GAMMA:-0.6}"
GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.00}"
GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-1.00}"
GRPO_RANDN_CLIP="${GRPO_RANDN_CLIP:-5.0}"
GRPO_DENOISED_CLIP="${GRPO_DENOISED_CLIP:-1.0}"
GRPO_MIN_SAMPLING_STD="${GRPO_MIN_SAMPLING_STD:-0.04}"
GRPO_MIN_LOGPROB_STD="${GRPO_MIN_LOGPROB_STD:-0.1}"
GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-16}"
GRPO_USE_BC_LOSS="${GRPO_USE_BC_LOSS:-true}"
GRPO_BC_COEFF="${GRPO_BC_COEFF:-0.10}"
GRPO_BC_ANNEAL="${GRPO_BC_ANNEAL:-true}"
GRPO_BC_COEFF_START="${GRPO_BC_COEFF_START:-0.10}"
GRPO_BC_COEFF_END="${GRPO_BC_COEFF_END:-0.05}"
GRPO_BC_ANNEAL_EPOCHS="${GRPO_BC_ANNEAL_EPOCHS:-5}"
GRPO_REFERENCE_KL_COEFF="${GRPO_REFERENCE_KL_COEFF:-0.02}"
SCORE_PROGRESS="${SCORE_PROGRESS:-10.0}"
SCORE_TTC="${SCORE_TTC:-5.0}"
SCORE_COMFORT="${SCORE_COMFORT:-2.0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export MASTER_ADDR
export MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=================================================="
echo "🚀 ReCogDrive Stage3 RL Single Node 8GPU (Ref-KL)"
echo "=================================================="
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "GPUs:        ${GPUS_PER_NODE}"
echo "Init ckpt:   ${INIT_CHECKPOINT}"
echo "Ref ckpt:    ${REFERENCE_POLICY_CHECKPOINT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Group size:  ${GRPO_SAMPLE_TIME}"
echo "BC coeff:    ${GRPO_BC_COEFF} (anneal=${GRPO_BC_ANNEAL}, end=${GRPO_BC_COEFF_END}, epochs=${GRPO_BC_ANNEAL_EPOCHS})"
echo "Ref KL coeff:${GRPO_REFERENCE_KL_COEFF}"
echo "Log file:    ${LOG_FILE}"
echo "Ckpt save:   top_k=${CKPT_SAVE_TOP_K} last=${CKPT_SAVE_LAST} every_n_epochs=${CKPT_EVERY_N_EPOCHS} monitor=${CKPT_MONITOR} mode=${CKPT_MODE}"
echo "Real eval:   enabled=${REAL_EVAL_ENABLED} split=${REAL_EVAL_SPLIT} top_k=${REAL_EVAL_TOP_K}"
echo "=================================================="

DL_ARGS=()
if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
  DL_ARGS+=("dataloader.params.prefetch_factor=${DATALOADER_PREFETCH_FACTOR}")
  DL_ARGS+=("dataloader.params.persistent_workers=${DATALOADER_PERSISTENT_WORKERS}")
else
  DL_ARGS+=("dataloader.params.prefetch_factor=null")
  DL_ARGS+=("dataloader.params.persistent_workers=false")
fi

EXTRA_OVERRIDE_ARGS=()
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES}"
fi

if [[ "${REAL_EVAL_ENABLED}" == "1" && "${REAL_EVAL_ASYNC_MODE}" == "tmux" ]]; then
  WATCHER_SESSION_NAME="${REAL_EVAL_SESSION_PREFIX}-$(basename "${OUTPUT_DIR}")"
  if tmux has-session -t "${WATCHER_SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${WATCHER_SESSION_NAME}"
  fi
  tmux new-session -d -s "${WATCHER_SESSION_NAME}" \
    "cd '${PROJECT_ROOT}' && \
     source /data/miniconda/etc/profile.d/conda.sh && \
     conda activate '${CONDA_ENV_NAME}' && \
     PROJECT_ROOT='${PROJECT_ROOT}' \
     RUNTIME_ROOT='${RUNTIME_ROOT}' \
     NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' \
     NAVSIM_OUTPUT_ROOT='${NAVSIM_OUTPUT_ROOT}' \
     NAVHARD_METRIC_CACHE_PATH='${NAVHARD_METRIC_CACHE_PATH}' \
     OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
     NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' \
     CONDA_ENV_NAME='${CONDA_ENV_NAME}' \
     RUN_DIR='${OUTPUT_DIR}' \
     TRAIN_TEST_SPLIT='${REAL_EVAL_SPLIT}' \
     POLL_INTERVAL_SEC='${REAL_EVAL_POLL_INTERVAL_SEC}' \
     REAL_EVAL_TOP_K='${REAL_EVAL_TOP_K}' \
     REAL_EVAL_KEEP_LAST='${REAL_EVAL_KEEP_LAST}' \
     REAL_EVAL_SCORE_DECIMALS='${REAL_EVAL_SCORE_DECIMALS}' \
     REAL_EVAL_GPUS='${REAL_EVAL_GPUS}' \
     SESSION_PREFIX='${REAL_EVAL_SESSION_PREFIX}' \
     bash '${REAL_EVAL_WATCHER_SCRIPT}'"
  echo "Watcher:     ${WATCHER_SESSION_NAME}"
fi

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py" \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type='single' \
  agent.grpo=True \
  +agent.rl_algo_type="${RL_ALGO}" \
  +agent.grpo_cfg.gamma_denoising="${GRPO_GAMMA}" \
  +agent.grpo_cfg.clip_advantage_lower_quantile="${GRPO_CLIP_LOW}" \
  +agent.grpo_cfg.clip_advantage_upper_quantile="${GRPO_CLIP_HIGH}" \
  +agent.grpo_cfg.randn_clip_value="${GRPO_RANDN_CLIP}" \
  +agent.grpo_cfg.denoised_clip_value="${GRPO_DENOISED_CLIP}" \
  +agent.grpo_cfg.min_sampling_denoising_std="${GRPO_MIN_SAMPLING_STD}" \
  +agent.grpo_cfg.min_logprob_denoising_std="${GRPO_MIN_LOGPROB_STD}" \
  +agent.grpo_cfg.sample_time="${GRPO_SAMPLE_TIME}" \
  +agent.grpo_cfg.use_bc_loss="${GRPO_USE_BC_LOSS}" \
  +agent.grpo_cfg.bc_coeff="${GRPO_BC_COEFF}" \
  +agent.grpo_cfg.bc_anneal="${GRPO_BC_ANNEAL}" \
  +agent.grpo_cfg.bc_coeff_start="${GRPO_BC_COEFF_START}" \
  +agent.grpo_cfg.bc_coeff_end="${GRPO_BC_COEFF_END}" \
  +agent.grpo_cfg.bc_anneal_epochs="${GRPO_BC_ANNEAL_EPOCHS}" \
  +agent.grpo_cfg.reference_kl_coeff="${GRPO_REFERENCE_KL_COEFF}" \
  +agent.grpo_cfg.scorer_config.progress_weight="${SCORE_PROGRESS}" \
  +agent.grpo_cfg.scorer_config.ttc_weight="${SCORE_TTC}" \
  +agent.grpo_cfg.scorer_config.comfortable_weight="${SCORE_COMFORT}" \
  agent.cache_hidden_state=True \
  agent.vlm_type="internvl" \
  agent.checkpoint_path="'${INIT_CHECKPOINT}'" \
  agent.reference_policy_checkpoint="'${REFERENCE_POLICY_CHECKPOINT}'" \
  agent.cache_loader_mode="${CACHE_LOADER_MODE}" \
  agent.dit_type="small" \
  agent.sampling_method="ddim" \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
  cache_loader_mode="${CACHE_LOADER_MODE}" \
  checkpoint.monitor="${CKPT_MONITOR}" \
  checkpoint.mode="${CKPT_MODE}" \
  checkpoint.save_top_k="${CKPT_SAVE_TOP_K}" \
  checkpoint.every_n_epochs="${CKPT_EVERY_N_EPOCHS}" \
  checkpoint.save_last="${CKPT_SAVE_LAST}" \
  checkpoint.filename="'${CKPT_FILENAME}'" \
  +auto_resume_latest=false \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  dataloader.params.batch_size="${DATALOADER_BATCH_SIZE}" \
  dataloader.params.num_workers="${DATALOADER_NUM_WORKERS}" \
  dataloader.params.pin_memory="${DATALOADER_PIN_MEMORY}" \
  "${DL_ARGS[@]}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS_PER_NODE}" \
  experiment_name=training_internvl_agent_dit_rl \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset="${USE_CACHE_WITHOUT_DATASET}" \
  force_cache_computation=False \
  worker=sequential \
  "${EXTRA_OVERRIDE_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1
