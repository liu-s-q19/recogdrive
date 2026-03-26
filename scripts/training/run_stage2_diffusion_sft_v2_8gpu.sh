#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"

source /data/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
MASTER_PORT="${MASTER_PORT:-63689}"
GPUS="${GPUS:-8}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${RUNTIME_ROOT}/exp}"
export NAVSIM_OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs}"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/dataset/navsim}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_descriptor}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export RUNTIME_TMP_BASE="${RUNTIME_TMP_BASE:-${TMPDIR}/stage2_v2}"
mkdir -p "${NAVSIM_EXP_ROOT}" "${NAVSIM_OUTPUT_ROOT}" "${TMPDIR}" "${RUNTIME_TMP_BASE}"

VLM_PATH="${VLM_PATH:-${PROJECT_ROOT}/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH="${CACHE_PATH:-/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train}"
TRAIN_ENTRY="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_ema.py"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_v2_8gpus_${TS}}"
LATEST_OUTPUT_LINK="${LATEST_OUTPUT_LINK:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_v2_8gpus_latest}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_stage2_v2_${TS}.log}"

mkdir -p "${OUTPUT_DIR}"
ln -sfn "${OUTPUT_DIR}" "${LATEST_OUTPUT_LINK}"

DL_ARGS=()
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  DL_ARGS+=("dataloader.params.prefetch_factor=${PREFETCH_FACTOR}")
  DL_ARGS+=("dataloader.params.persistent_workers=${PERSISTENT_WORKERS}")
else
  DL_ARGS+=("dataloader.params.prefetch_factor=null")
  DL_ARGS+=("dataloader.params.persistent_workers=false")
fi

echo "=================================================="
echo "ReCogDrive Stage-2 Diffusion SFT v2"
echo "=================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Runtime root: ${RUNTIME_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Latest link:  ${LATEST_OUTPUT_LINK}"
echo "Cache path:   ${CACHE_PATH}"
echo "Checkpoint VLM: ${VLM_PATH}"
echo "GPUs:         ${GPUS}"
echo "Master port:  ${MASTER_PORT}"
echo "Log file:     ${LOG_FILE}"
echo "=================================================="

torchrun \
  --standalone \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS}" \
  "${TRAIN_ENTRY}" \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type='single' \
  agent.cache_hidden_state=True \
  agent.vlm_type="internvl" \
  agent.dit_type="small" \
  agent.sampling_method="ddim" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.num_nodes=1 \
  trainer.params.limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  trainer.params.limit_val_batches="${LIMIT_VAL_BATCHES}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.pin_memory="${PIN_MEMORY}" \
  "${DL_ARGS[@]}" \
  experiment_name=training_internvl_agent_dit_v2 \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  cache_loader_mode=legacy_cached_features \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  worker=sequential \
  > "${LOG_FILE}" 2>&1
