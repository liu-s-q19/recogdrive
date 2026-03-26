#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-epoch=*.ckpt}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
MASTER_PORT="${MASTER_PORT:-29751}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh}"
SESSION_PREFIX="${SESSION_PREFIX:-eval-navhard-rpp}"
REAL_EVAL_TOP_K="${REAL_EVAL_TOP_K:-3}"
REAL_EVAL_KEEP_LAST="${REAL_EVAL_KEEP_LAST:-1}"
REAL_EVAL_SCORE_DECIMALS="${REAL_EVAL_SCORE_DECIMALS:-6}"
REAL_EVAL_GPUS="${REAL_EVAL_GPUS:-8}"
REGISTRY_PATH="${REGISTRY_PATH:-${RUN_DIR}/navhard_eval_registry.json}"
RANKING_PATH="${RANKING_PATH:-${RUN_DIR}/navhard_eval_ranking.json}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/navsimv2-recogdrive/bin/python}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${RUN_DIR}/auto_eval_navhard}"
CHECKPOINT_SEARCH_ROOT="${CHECKPOINT_SEARCH_ROOT:-${RUN_DIR}/lightning_logs}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/dataset/navsim}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/dataset/navsim/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${RUNTIME_ROOT}/exp}"
NAVSIM_OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs}"
NAVHARD_METRIC_CACHE_PATH="${NAVHARD_METRIC_CACHE_PATH:-/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733}"
KEEP_LAST_FLAG=""
if [[ "${REAL_EVAL_KEEP_LAST}" == "1" ]]; then
  KEEP_LAST_FLAG="--keep-last"
fi

mkdir -p "${EVAL_LOG_ROOT}"

echo "[watcher] run_dir=${RUN_DIR}"
echo "[watcher] checkpoint_glob=${CHECKPOINT_GLOB}"
echo "[watcher] train_test_split=${TRAIN_TEST_SPLIT}"
echo "[watcher] registry_path=${REGISTRY_PATH}"
echo "[watcher] ranking_path=${RANKING_PATH}"

while true; do
  if [[ ! -d "${CHECKPOINT_SEARCH_ROOT}" ]]; then
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  mapfile -t checkpoint_paths < <(find "${CHECKPOINT_SEARCH_ROOT}" -path "*/checkpoints/${CHECKPOINT_GLOB}" -type f | sort)

  for checkpoint_path in "${checkpoint_paths[@]}"; do
    checkpoint_name="$(basename "${checkpoint_path}" .ckpt)"
    eval_dir="${EVAL_LOG_ROOT}/${checkpoint_name}"
    session_name="${SESSION_PREFIX}-${checkpoint_name}"

    register_result="$("${PYTHON_BIN}" -m navsim.planning.script.navhard_async_eval \
      register-checkpoint \
      --registry-path "${REGISTRY_PATH}" \
      --checkpoint-path "${checkpoint_path}" \
      --eval-dir "${eval_dir}" \
      --session-name "${session_name}")"
    if [[ "${register_result}" != "registered" ]]; then
      continue
    fi

    mkdir -p "${eval_dir}"

    if tmux has-session -t "${session_name}" 2>/dev/null; then
      tmux kill-session -t "${session_name}"
    fi

    tmux new-session -d -s "${session_name}" \
      "cd '${PROJECT_ROOT}' && \
       source /data/miniconda/etc/profile.d/conda.sh && \
       conda activate '${CONDA_ENV_NAME}' && \
       export PROJECT_ROOT='${PROJECT_ROOT}' \
              RUNTIME_ROOT='${RUNTIME_ROOT}' \
              NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' \
              NAVSIM_OUTPUT_ROOT='${NAVSIM_OUTPUT_ROOT}' \
              NAVHARD_METRIC_CACHE_PATH='${NAVHARD_METRIC_CACHE_PATH}' \
              METRIC_CACHE_PATH='${NAVHARD_METRIC_CACHE_PATH}' \
              OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
              NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' \
              TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' \
              GPUS='${REAL_EVAL_GPUS}' \
              MASTER_PORT='${MASTER_PORT}' && \
       CHECKPOINT='${checkpoint_path}' \
       OUTPUT_DIR='${eval_dir}' \
       LOG_FILE='${eval_dir}/eval_tmux.log' \
       bash '${EVAL_SCRIPT}'; \
       status=\$?; \
       if [[ \$status -eq 0 && -f '${eval_dir}/summary.json' ]]; then \
         '${PYTHON_BIN}' -m navsim.planning.script.navhard_async_eval complete-eval \
           --run-dir '${RUN_DIR}' \
           --registry-path '${REGISTRY_PATH}' \
           --ranking-path '${RANKING_PATH}' \
           --checkpoint-path '${checkpoint_path}' \
           --summary-path '${eval_dir}/summary.json' \
           --status succeeded \
           --top-k '${REAL_EVAL_TOP_K}' \
           --score-decimals '${REAL_EVAL_SCORE_DECIMALS}' \
           ${KEEP_LAST_FLAG}; \
       else \
         '${PYTHON_BIN}' -m navsim.planning.script.navhard_async_eval complete-eval \
           --run-dir '${RUN_DIR}' \
           --registry-path '${REGISTRY_PATH}' \
           --ranking-path '${RANKING_PATH}' \
           --checkpoint-path '${checkpoint_path}' \
           --summary-path '${eval_dir}/summary.json' \
           --status failed \
           --top-k '${REAL_EVAL_TOP_K}' \
           --score-decimals '${REAL_EVAL_SCORE_DECIMALS}' \
           ${KEEP_LAST_FLAG}; \
       fi"
    echo "[watcher] launched session=${session_name} checkpoint=${checkpoint_path}"
  done

  sleep "${POLL_INTERVAL_SEC}"
done
