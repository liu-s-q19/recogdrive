#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-epoch=9-step=*.ckpt}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
MASTER_PORT="${MASTER_PORT:-29751}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/lightning_logs/version_0/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_DIR}/auto_eval_epoch9}"
ENTRYPOINT="${ENTRYPOINT:-${PROJECT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py}"
EXP_NAME_OVERRIDE="${EXP_NAME_OVERRIDE:-$(basename "${RUN_DIR}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXP_NAME_OVERRIDE}_epoch9_navtest}"

mkdir -p "${EVAL_ROOT}"

echo "[watcher] run_dir=${RUN_DIR}"
echo "[watcher] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[watcher] checkpoint_glob=${CHECKPOINT_GLOB}"
echo "[watcher] eval_root=${EVAL_ROOT}"

while true; do
  checkpoint_path="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "${CHECKPOINT_GLOB}" | sort | tail -n 1 || true)"
  if [[ -n "${checkpoint_path}" ]]; then
    echo "[watcher] found checkpoint=${checkpoint_path}"
    break
  fi
  echo "[watcher] checkpoint not ready, sleep ${POLL_INTERVAL_SEC}s"
  sleep "${POLL_INTERVAL_SEC}"
done

timestamp="$(date -u +%Y%m%d_%H%M%S)"
eval_dir="${EVAL_ROOT}/eval_epoch9_${timestamp}"
mkdir -p "${eval_dir}"

cd "${PROJECT_ROOT}"
MASTER_PORT="${MASTER_PORT}" \
CHECKPOINT="${checkpoint_path}" \
OUTPUT_DIR="${eval_dir}" \
LOG_FILE="${eval_dir}/eval_tmux.log" \
EXP_NAME_OVERRIDE="${EXP_NAME_OVERRIDE}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
ENTRYPOINT="${ENTRYPOINT}" \
bash "${PROJECT_ROOT}/scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh"

echo "[watcher] eval finished output_dir=${eval_dir}"
