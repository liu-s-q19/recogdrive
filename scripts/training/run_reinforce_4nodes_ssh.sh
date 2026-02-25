#!/usr/bin/env bash
set -euo pipefail

# 4-node SSH launcher for ReCogDrive Stage3 RL (reinforce)
# - Verifies /dev/shm=128G on every node
# - Launches worker nodes (rank 1..3) via SSH + nohup
# - Launches master rank0 locally (foreground or background)

# ----------------- SSH / cluster -----------------
SSH_PORT="${SSH_PORT:-2289}"
SSH_USER="${SSH_USER:-root}"

NODES=(
  "10.199.7.32"  # rank 0
  "10.199.7.33"  # rank 1
  "10.199.7.190" # rank 2
  "10.199.7.191" # rank 3
)

NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.199.7.32}"
MASTER_PORT="${MASTER_PORT:-29520}"

# Modes
# - QUICK_VALIDATE=1 : run a light smoke config
# - RUN_MASTER_BG=1  : run rank0 with nohup and exit after checks
# - AUTO_CHECK=1     : when RUN_MASTER_BG=1, validate pids/logs automatically
QUICK_VALIDATE="${QUICK_VALIDATE:-0}"
RUN_MASTER_BG="${RUN_MASTER_BG:-0}"
AUTO_CHECK="${AUTO_CHECK:-1}"
CHECK_TIMEOUT_SEC="${CHECK_TIMEOUT_SEC:-300}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-12}"
ERROR_PATTERN="${ERROR_PATTERN:-RuntimeError|Exception:|NCCL error|Address already in use|CUDA out of memory|Segmentation fault|torch.distributed.elastic.multiprocessing.errors.ChildFailedError|No space left on device|Bus error}"

# Log cleanup
CLEAN_OLD_LOGS="${CLEAN_OLD_LOGS:-1}"
LOG_KEEP_RUNS="${LOG_KEEP_RUNS:-2}"

# Network tuning
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond4}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond4}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

# ----------------- Project / training -----------------
PROJECT_ROOT="${PROJECT_ROOT:-/data/liushiqi/recogdrive}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/liushiqi/recogdrive/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/liushiqi/recogdrive/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/data/liushiqi/recogdrive}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/liushiqi/recogdrive/dataset/navsim}"

VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_train}"

# checkpoint auto resolve (prefer EMA)
CHECKPOINT="${CHECKPOINT:-/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints}"
if [[ -d "${CHECKPOINT}" ]]; then
  DEFAULT_CKPT_DIR="${CHECKPOINT}"
else
  DEFAULT_CKPT_DIR="$(dirname "${CHECKPOINT}")"
fi
DEFAULT_CKPT_EMA="${DEFAULT_CKPT_DIR}/last-EMA.ckpt"
DEFAULT_CKPT_RAW="${DEFAULT_CKPT_DIR}/last.ckpt"
if [[ -f "${CHECKPOINT}" ]]; then
  :
elif [[ -f "${DEFAULT_CKPT_EMA}" ]]; then
  CHECKPOINT="${DEFAULT_CKPT_EMA}"
elif [[ -f "${DEFAULT_CKPT_RAW}" ]]; then
  CHECKPOINT="${DEFAULT_CKPT_RAW}"
else
  echo "[ERROR] No checkpoint file found."
  echo "        Input CHECKPOINT: ${CHECKPOINT}"
  echo "        Tried: ${DEFAULT_CKPT_EMA}"
  echo "        Tried: ${DEFAULT_CKPT_RAW}"
  exit 1
fi

RL_ALGO="${RL_ALGO:-reinforce}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/liushiqi/recogdrive/outputs/recogdrive_stage3_rl_reinforce_4nodes_32gpus}"

# /dev/shm stability for dataloader
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_descriptor}"
TMP_BASEDIR="${TMP_BASEDIR:-$NAVSIM_EXP_ROOT/tmp}"
CLEAN_TMP_BEFORE_RUN="${CLEAN_TMP_BEFORE_RUN:-1}"

DATALOADER_BATCH_SIZE="${DATALOADER_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"

MAX_EPOCHS="${MAX_EPOCHS:-10}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"

# GRPO / reinforce hyperparameters
GRPO_GAMMA="${GRPO_GAMMA:-0.6}"
GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.00}"
GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-1.00}"
GRPO_RANDN_CLIP="${GRPO_RANDN_CLIP:-5.0}"
GRPO_DENOISED_CLIP="${GRPO_DENOISED_CLIP:-1.0}"
GRPO_MIN_SAMPLING_STD="${GRPO_MIN_SAMPLING_STD:-0.04}"
GRPO_MIN_LOGPROB_STD="${GRPO_MIN_LOGPROB_STD:-0.1}"
SCORE_PROGRESS="${SCORE_PROGRESS:-10.0}"
SCORE_TTC="${SCORE_TTC:-5.0}"
SCORE_COMFORT="${SCORE_COMFORT:-2.0}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

CONDA_SH="${CONDA_SH:-/data/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-navsim}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py}"

SSH_OPTS=(
  -p "${SSH_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ConnectTimeout=8
)

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

cleanup_old_runs() {
  local out_dir="$1"
  local keep_runs="$2"

  [[ -d "${out_dir}" ]] || return 0

  mapfile -t all_ts < <(
    ls -1 "${out_dir}"/train_rank0_*.log 2>/dev/null \
      | sed -E 's#.*train_rank0_([0-9]{8}_[0-9]{6})\.log#\1#' \
      | sort -u -r
  )

  if (( ${#all_ts[@]} <= keep_runs )); then
    return 0
  fi

  echo "[CLEAN] Found ${#all_ts[@]} runs in ${out_dir}, keeping latest ${keep_runs}"
  for old_ts in "${all_ts[@]:keep_runs}"; do
    echo "[CLEAN] Removing logs/artifacts for run TS=${old_ts}"
    rm -f "${out_dir}"/train_rank*_${old_ts}.log || true
    rm -f "${out_dir}"/env_debug_rank*_${old_ts}.log || true
    rm -f "${out_dir}"/pid_rank*_${old_ts}.txt || true
  done
}

if [[ "${QUICK_VALIDATE}" == "1" ]]; then
  GPUS_PER_NODE="1"
  DATALOADER_BATCH_SIZE="1"
  DATALOADER_NUM_WORKERS="1"
  DATALOADER_PIN_MEMORY="false"
  DATALOADER_PERSISTENT_WORKERS="false"
  MAX_EPOCHS="1"
  OUTPUT_DIR="/data/liushiqi/recogdrive/outputs/recogdrive_reinforce_quick_validate_4nodes_${TS}"
fi

echo "=================================================="
echo "🚀 ReCogDrive Stage3 RL (reinforce) 4-node SSH"
echo "=================================================="
echo "Nodes:       ${NODES[*]}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "NNODES:      ${NNODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "Conda env:   ${CONDA_ENV}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Metric cache:${METRIC_CACHE_PATH}"
echo "NCCL_IFNAME: ${NCCL_SOCKET_IFNAME} (IB disabled=${NCCL_IB_DISABLE})"
echo "TS:          ${TS}"
echo "Max epochs:  ${MAX_EPOCHS}"
echo "Quick mode:  ${QUICK_VALIDATE}"
echo "Master BG:   ${RUN_MASTER_BG}"
echo "Auto check:  ${AUTO_CHECK}"
echo "Clean logs:  ${CLEAN_OLD_LOGS} (keep=${LOG_KEEP_RUNS})"
echo "=================================================="

mkdir -p "${OUTPUT_DIR}"
if [[ "${CLEAN_OLD_LOGS}" == "1" ]]; then
  cleanup_old_runs "${OUTPUT_DIR}" "${LOG_KEEP_RUNS}"
fi

# ----------------- Step 0: ensure /dev/shm=128G everywhere -----------------
echo "[STEP] Ensure /dev/shm is 128G on all nodes"
for node in "${NODES[@]}"; do
  echo "==> ${node}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node}" "mount -o remount,size=128G /dev/shm || true; df -h /dev/shm; mount | grep 'on /dev/shm' || true"
done

# ----------------- Step 1: launch workers -----------------
echo "[STEP] Launch worker ranks 1..3 via SSH (nohup)"
for node_rank in 1 2 3; do
  node_ip="${NODES[$node_rank]}"
  echo "==> worker rank=${node_rank} node=${node_ip}"

  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" \
    "MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' NNODES='${NNODES}' GPUS_PER_NODE='${GPUS_PER_NODE}' NODE_RANK='${node_rank}' TS='${TS}' \
     PROJECT_ROOT='${PROJECT_ROOT}' CONDA_SH='${CONDA_SH}' CONDA_ENV='${CONDA_ENV}' TRAIN_ENTRY='${TRAIN_ENTRY}' \
     TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' \
     NAVSIM_DEVKIT_ROOT='${NAVSIM_DEVKIT_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
     CACHE_PATH='${CACHE_PATH}' METRIC_CACHE_PATH='${METRIC_CACHE_PATH}' VLM_PATH='${VLM_PATH}' CHECKPOINT='${CHECKPOINT}' OUTPUT_DIR='${OUTPUT_DIR}' \
     RL_ALGO='${RL_ALGO}' MAX_EPOCHS='${MAX_EPOCHS}' EXTRA_OVERRIDES='${EXTRA_OVERRIDES}' \
     GRPO_GAMMA='${GRPO_GAMMA}' GRPO_CLIP_LOW='${GRPO_CLIP_LOW}' GRPO_CLIP_HIGH='${GRPO_CLIP_HIGH}' \
     GRPO_RANDN_CLIP='${GRPO_RANDN_CLIP}' GRPO_DENOISED_CLIP='${GRPO_DENOISED_CLIP}' \
     GRPO_MIN_SAMPLING_STD='${GRPO_MIN_SAMPLING_STD}' GRPO_MIN_LOGPROB_STD='${GRPO_MIN_LOGPROB_STD}' \
     SCORE_PROGRESS='${SCORE_PROGRESS}' SCORE_TTC='${SCORE_TTC}' SCORE_COMFORT='${SCORE_COMFORT}' \
     DATALOADER_BATCH_SIZE='${DATALOADER_BATCH_SIZE}' DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' \
     DATALOADER_PREFETCH_FACTOR='${DATALOADER_PREFETCH_FACTOR}' DATALOADER_PIN_MEMORY='${DATALOADER_PIN_MEMORY}' \
     DATALOADER_PERSISTENT_WORKERS='${DATALOADER_PERSISTENT_WORKERS}' \
     DISABLE_TORCH_COMPILE='${DISABLE_TORCH_COMPILE}' TORCH_SHARING_STRATEGY='${TORCH_SHARING_STRATEGY}' \
     TMP_BASEDIR='${TMP_BASEDIR}' CLEAN_TMP_BEFORE_RUN='${CLEAN_TMP_BEFORE_RUN}' \
     NCCL_IB_DISABLE='${NCCL_IB_DISABLE}' NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' \
     NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}' NCCL_P2P_DISABLE='${NCCL_P2P_DISABLE}' NCCL_SHM_DISABLE='${NCCL_SHM_DISABLE}' \
     TORCH_DISTRIBUTED_DEBUG='${TORCH_DISTRIBUTED_DEBUG}' bash -s" <<'REMOTE'
set -euo pipefail

mount -o remount,size=128G /dev/shm || true

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

NODE_TMPDIR="${TMP_BASEDIR}/${TS}/rank${NODE_RANK}"
if [[ "${CLEAN_TMP_BEFORE_RUN}" == "1" ]]; then
  rm -rf "${NODE_TMPDIR}" || true
fi
mkdir -p "${NODE_TMPDIR}" "${OUTPUT_DIR}"

printenv > "${OUTPUT_DIR}/env_debug_rank${NODE_RANK}_${TS}.log" || true

ulimit -n 65535 || true
ulimit -l unlimited || true

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MASTER_ADDR MASTER_PORT
export TORCH_SHARING_STRATEGY
export NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export NCCL_SOCKET_FAMILY NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export TORCH_DISTRIBUTED_DEBUG DISABLE_TORCH_COMPILE

LOG_FILE="${OUTPUT_DIR}/train_rank${NODE_RANK}_${TS}.log"

echo "[START] worker rank=${NODE_RANK} host=$(hostname) log=${LOG_FILE}"

DL_ARGS=()
if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
  DL_ARGS+=("dataloader.params.prefetch_factor=${DATALOADER_PREFETCH_FACTOR}")
  DL_ARGS+=("dataloader.params.persistent_workers=${DATALOADER_PERSISTENT_WORKERS}")
else
  DL_ARGS+=("dataloader.params.prefetch_factor=null")
  DL_ARGS+=("dataloader.params.persistent_workers=false")
fi

nohup torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${TRAIN_ENTRY}" \
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
  +agent.grpo_cfg.scorer_config.progress_weight="${SCORE_PROGRESS}" \
  +agent.grpo_cfg.scorer_config.ttc_weight="${SCORE_TTC}" \
  +agent.grpo_cfg.scorer_config.comfortable_weight="${SCORE_COMFORT}" \
  agent.cache_hidden_state=True \
  agent.vlm_type="internvl" \
  agent.checkpoint_path="${CHECKPOINT}" \
  agent.reference_policy_checkpoint="${CHECKPOINT}" \
  agent.dit_type="small" \
  agent.sampling_method="ddim" \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
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
  use_cache_without_dataset=True \
  force_cache_computation=False \
  worker=sequential \
  ${EXTRA_OVERRIDES} \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${OUTPUT_DIR}/pid_rank${NODE_RANK}_${TS}.txt" || true
echo "[LAUNCHED] worker rank=${NODE_RANK} pid=$!"
REMOTE
done

# ----------------- Step 2: launch master rank0 -----------------
echo "[STEP] Launch master rank0"
mount -o remount,size=128G /dev/shm || true
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

NODE_TMPDIR="${TMP_BASEDIR}/${TS}/rank0"
if [[ "${CLEAN_TMP_BEFORE_RUN}" == "1" ]]; then
  rm -rf "${NODE_TMPDIR}" || true
fi
mkdir -p "${NODE_TMPDIR}" "${OUTPUT_DIR}"

printenv > "${OUTPUT_DIR}/env_debug_rank0_${TS}.log" || true

ulimit -n 65535 || true
ulimit -l unlimited || true

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MASTER_ADDR MASTER_PORT
export TORCH_SHARING_STRATEGY
export DISABLE_TORCH_COMPILE

LOG_FILE="${OUTPUT_DIR}/train_rank0_${TS}.log"

echo "[START] master rank=0 host=$(hostname) log=${LOG_FILE}"

DL_ARGS=()
if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
  DL_ARGS+=("dataloader.params.prefetch_factor=${DATALOADER_PREFETCH_FACTOR}")
  DL_ARGS+=("dataloader.params.persistent_workers=${DATALOADER_PERSISTENT_WORKERS}")
else
  DL_ARGS+=("dataloader.params.prefetch_factor=null")
  DL_ARGS+=("dataloader.params.persistent_workers=false")
fi

MASTER_CMD=(torchrun
  --nnodes="${NNODES}"
  --node_rank=0
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  --nproc_per_node="${GPUS_PER_NODE}"
  "${TRAIN_ENTRY}"
  agent=recogdrive_agent
  agent.lr=1e-4
  agent.vlm_path="${VLM_PATH}"
  agent.cam_type='single'
  agent.grpo=True
  +agent.rl_algo_type="${RL_ALGO}"
  +agent.grpo_cfg.gamma_denoising="${GRPO_GAMMA}"
  +agent.grpo_cfg.clip_advantage_lower_quantile="${GRPO_CLIP_LOW}"
  +agent.grpo_cfg.clip_advantage_upper_quantile="${GRPO_CLIP_HIGH}"
  +agent.grpo_cfg.randn_clip_value="${GRPO_RANDN_CLIP}"
  +agent.grpo_cfg.denoised_clip_value="${GRPO_DENOISED_CLIP}"
  +agent.grpo_cfg.min_sampling_denoising_std="${GRPO_MIN_SAMPLING_STD}"
  +agent.grpo_cfg.min_logprob_denoising_std="${GRPO_MIN_LOGPROB_STD}"
  +agent.grpo_cfg.scorer_config.progress_weight="${SCORE_PROGRESS}"
  +agent.grpo_cfg.scorer_config.ttc_weight="${SCORE_TTC}"
  +agent.grpo_cfg.scorer_config.comfortable_weight="${SCORE_COMFORT}"
  agent.cache_hidden_state=True
  agent.vlm_type="internvl"
  agent.checkpoint_path="${CHECKPOINT}"
  agent.reference_policy_checkpoint="${CHECKPOINT}"
  agent.dit_type="small"
  agent.sampling_method="ddim"
  agent.metric_cache_path="${METRIC_CACHE_PATH}"
  trainer.params.max_epochs="${MAX_EPOCHS}"
  dataloader.params.batch_size="${DATALOADER_BATCH_SIZE}"
  dataloader.params.num_workers="${DATALOADER_NUM_WORKERS}"
  dataloader.params.pin_memory="${DATALOADER_PIN_MEMORY}"
  "${DL_ARGS[@]}"
  trainer.params.num_nodes="${NNODES}"
  trainer.params.devices="${GPUS_PER_NODE}"
  experiment_name=training_internvl_agent_dit_rl
  train_test_split="${TRAIN_TEST_SPLIT}"
  cache_path="${CACHE_PATH}"
  output_dir="${OUTPUT_DIR}"
  use_cache_without_dataset=True
  force_cache_computation=False
  worker=sequential
  ${EXTRA_OVERRIDES})

if [[ "${RUN_MASTER_BG}" == "1" ]]; then
  nohup "${MASTER_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo $! > "${OUTPUT_DIR}/pid_rank0_${TS}.txt" || true
  echo "[LAUNCHED] master rank=0 pid=$!"
  echo "Logs: ${OUTPUT_DIR}/train_rank*_${TS}.log"

  if [[ "${AUTO_CHECK}" == "1" ]]; then
    echo "[CHECK] Waiting for multi-node processes/logs to become healthy..."
    deadline=$((SECONDS + CHECK_TIMEOUT_SEC))
    all_healthy=0

    while (( SECONDS < deadline )); do
      all_healthy=1

      if [[ -f "${OUTPUT_DIR}/pid_rank0_${TS}.txt" ]]; then
        master_pid="$(cat "${OUTPUT_DIR}/pid_rank0_${TS}.txt" 2>/dev/null || true)"
      else
        master_pid=""
      fi

      if [[ -z "${master_pid}" ]] || ! kill -0 "${master_pid}" 2>/dev/null; then
        all_healthy=0
      fi
      if [[ -f "${OUTPUT_DIR}/train_rank0_${TS}.log" ]] && grep -Eiq "${ERROR_PATTERN}" "${OUTPUT_DIR}/train_rank0_${TS}.log"; then
        echo "[CHECK][FAIL] rank0 log contains error pattern: ${ERROR_PATTERN}"
        tail -n 60 "${OUTPUT_DIR}/train_rank0_${TS}.log" || true
        exit 1
      fi

      for node_rank in 1 2 3; do
        node_ip="${NODES[$node_rank]}"
        if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" \
          "set -euo pipefail; pid_file='${OUTPUT_DIR}/pid_rank${node_rank}_${TS}.txt'; log_file='${OUTPUT_DIR}/train_rank${node_rank}_${TS}.log'; \
           [[ -f \"\${pid_file}\" ]] || exit 10; \
           pid=\$(cat \"\${pid_file}\" 2>/dev/null || true); \
           [[ -n \"\${pid}\" ]] || exit 11; \
           kill -0 \"\${pid}\" 2>/dev/null || exit 11; \
           if [[ -f \"\${log_file}\" ]] && grep -Eiq '${ERROR_PATTERN}' \"\${log_file}\"; then exit 12; fi"; then
          rc=$?
          if [[ $rc -eq 12 ]]; then
            echo "[CHECK][FAIL] rank${node_rank} (${node_ip}) log contains error pattern: ${ERROR_PATTERN}"
            ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" "tail -n 60 '${OUTPUT_DIR}/train_rank${node_rank}_${TS}.log' || true"
            exit 1
          fi
          all_healthy=0
        fi
      done

      if [[ "${all_healthy}" -eq 1 ]]; then
        echo "[CHECK][OK] All ranks are alive, and no obvious fatal errors were detected in logs."
        echo "[CHECK] Log tail rank0:" && tail -n 20 "${OUTPUT_DIR}/train_rank0_${TS}.log" || true
        for node_rank in 1 2 3; do
          node_ip="${NODES[$node_rank]}"
          echo "[CHECK] Log tail rank${node_rank} (${node_ip}):"
          ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" "tail -n 20 '${OUTPUT_DIR}/train_rank${node_rank}_${TS}.log' || true"
        done
        break
      fi

      sleep "${CHECK_INTERVAL_SEC}"
    done

    if [[ "${all_healthy}" -ne 1 ]]; then
      echo "[CHECK][WARN] Timeout (${CHECK_TIMEOUT_SEC}s): some ranks may still be initializing."
      echo "You can inspect: ${OUTPUT_DIR}/train_rank*_${TS}.log"
    fi
  fi

  exit 0
else
  "${MASTER_CMD[@]}" > "${LOG_FILE}" 2>&1
fi

echo "[DONE] master finished"
echo "Logs: ${OUTPUT_DIR}/train_rank*_${TS}.log"
