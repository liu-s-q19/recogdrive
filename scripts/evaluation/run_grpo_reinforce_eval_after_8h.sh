#!/usr/bin/env bash
set -euo pipefail

# Wait 8 hours (default), then evaluate GRPO + REINFORCE epoch=9 checkpoints concurrently.
# Default parallel mode uses 4 GPUs + 4 GPUs on a single 8-GPU machine.

SLEEP_HOURS="${SLEEP_HOURS:-8}"
WAIT_EPOCH9_TIMEOUT_HOURS="${WAIT_EPOCH9_TIMEOUT_HOURS:-6}"
POLL_SECONDS="${POLL_SECONDS:-120}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/liushiqi/recogdrive}"
CONDA_SH="${CONDA_SH:-/data/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-navsim}"

GRPO_OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/recogdrive_stage3_rl_grpov2_4nodes}"
REINFORCE_OUTPUT_ROOT="${REINFORCE_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/recogdrive_stage3_rl_reinforce_4nodes_v2}"

VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH_EVAL="${CACHE_PATH_EVAL:-$PROJECT_ROOT/exp/recogdrive_agent_cache_dir_train_test}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"

GRPO_DEVICES="${GRPO_DEVICES:-0,1,2,3}"
REINFORCE_DEVICES="${REINFORCE_DEVICES:-4,5,6,7}"
MASTER_PORT_GRPO="${MASTER_PORT_GRPO:-63669}"
MASTER_PORT_REINFORCE="${MASTER_PORT_REINFORCE:-63679}"

AGENT_GRPO_FLAG="${AGENT_GRPO_FLAG:-true}"
AGENT_CACHE_HIDDEN_STATE="${AGENT_CACHE_HIDDEN_STATE:-true}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$PROJECT_ROOT/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/dataset/navsim}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

ts="$(date +%Y%m%d_%H%M%S)"

count_csv_items() {
  local csv="$1"
  awk -F',' '{print NF}' <<< "${csv}"
}

check_no_overlap() {
  local a_csv="$1"
  local b_csv="$2"
  local a
  local b
  IFS=',' read -r -a a <<< "${a_csv}"
  IFS=',' read -r -a b <<< "${b_csv}"
  for x in "${a[@]}"; do
    for y in "${b[@]}"; do
      if [[ "${x}" == "${y}" ]]; then
        echo "[ERROR] GPU overlap detected: ${x}"
        echo "        GRPO_DEVICES=${a_csv}"
        echo "        REINFORCE_DEVICES=${b_csv}"
        exit 1
      fi
    done
  done
}

latest_version_dir() {
  local root="$1"
  local latest
  latest="$(find "${root}/lightning_logs" -maxdepth 1 -mindepth 1 -type d -name 'version_*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s\n' "${latest}"
}

resolve_epoch9_ckpt() {
  local output_root="$1"
  local ver_dir
  local ckpt

  ver_dir="$(latest_version_dir "${output_root}" || true)"
  if [[ -z "${ver_dir}" ]]; then
    return 1
  fi

  ckpt="$(find "${ver_dir}/checkpoints" -maxdepth 1 -type f -name 'epoch=9-*.ckpt' | sort -V | tail -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    return 1
  fi
  printf '%s\n' "${ckpt}"
}

wait_epoch9_ckpt() {
  local name="$1"
  local output_root="$2"
  local timeout_sec="$3"
  local elapsed=0
  local ckpt

  while (( elapsed <= timeout_sec )); do
    ckpt="$(resolve_epoch9_ckpt "${output_root}" || true)"
    if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
      printf '%s\n' "${ckpt}"
      return 0
    fi
    echo "[WAIT] ${name}: epoch=9 checkpoint not found yet under ${output_root}, retry in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    elapsed=$((elapsed + POLL_SECONDS))
  done

  echo "[ERROR] ${name}: timeout waiting epoch=9 checkpoint (timeout=${timeout_sec}s)"
  return 1
}

run_one_eval_bg() {
  local tag="$1"
  local checkpoint="$2"
  local devices="$3"
  local master_port="$4"

  local gpus
  gpus="$(count_csv_items "${devices}")"

  local checkpoint_hydra
  checkpoint_hydra="${checkpoint//=/\\=}"

  local exp_name
  exp_name="${tag}_epoch9_${ts}"

  local eval_root
  eval_root="${PROJECT_ROOT}/outputs/${tag}/eval_epoch9_${ts}"
  mkdir -p "${eval_root}"

  local log_file
  log_file="${eval_root}/eval_${tag}.log"

  echo "[LAUNCH] ${tag} eval" >&2
  echo "         checkpoint=${checkpoint}" >&2
  echo "         devices=${devices} (gpus=${gpus})" >&2
  echo "         master_port=${master_port}" >&2
  echo "         log=${log_file}" >&2

  CUDA_VISIBLE_DEVICES="${devices}" torchrun \
    --nproc_per_node="${gpus}" \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="${checkpoint_hydra}" \
    agent.vlm_path="${VLM_PATH}" \
    agent.cam_type='single' \
    agent.grpo="${AGENT_GRPO_FLAG}" \
    agent.cache_hidden_state="${AGENT_CACHE_HIDDEN_STATE}" \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    cache_path="${CACHE_PATH_EVAL}" \
    use_cache_without_dataset=True \
    agent.sampling_method="${SAMPLE_METHOD}" \
    worker=sequential \
    output_dir="${eval_root}" \
    experiment_name="${exp_name}" \
    > "${log_file}" 2>&1 &

  echo $!
}

main() {
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  cd "${PROJECT_ROOT}"

  check_no_overlap "${GRPO_DEVICES}" "${REINFORCE_DEVICES}"

  local sleep_sec
  sleep_sec="$((SLEEP_HOURS * 3600))"
  local wait_timeout_sec
  wait_timeout_sec="$((WAIT_EPOCH9_TIMEOUT_HOURS * 3600))"

  echo "=================================================="
  echo "[PLAN] Wait then dual-evaluate epoch=9"
  echo "SLEEP_HOURS=${SLEEP_HOURS}, WAIT_EPOCH9_TIMEOUT_HOURS=${WAIT_EPOCH9_TIMEOUT_HOURS}"
  echo "GRPO_OUTPUT_ROOT=${GRPO_OUTPUT_ROOT}"
  echo "REINFORCE_OUTPUT_ROOT=${REINFORCE_OUTPUT_ROOT}"
  echo "GRPO_DEVICES=${GRPO_DEVICES}, REINFORCE_DEVICES=${REINFORCE_DEVICES}"
  echo "MASTER_PORT_GRPO=${MASTER_PORT_GRPO}, MASTER_PORT_REINFORCE=${MASTER_PORT_REINFORCE}"
  echo "=================================================="

  if (( sleep_sec > 0 )); then
    echo "[WAIT] sleeping ${SLEEP_HOURS} hour(s) before evaluation..."
    sleep "${sleep_sec}"
  fi

  local grpo_ckpt
  local reinforce_ckpt
  grpo_ckpt="$(wait_epoch9_ckpt "GRPO" "${GRPO_OUTPUT_ROOT}" "${wait_timeout_sec}")"
  reinforce_ckpt="$(wait_epoch9_ckpt "REINFORCE" "${REINFORCE_OUTPUT_ROOT}" "${wait_timeout_sec}")"

  local pid_grpo
  local pid_reinforce
  pid_grpo="$(run_one_eval_bg "recogdrive_stage3_rl_grpov2_4nodes" "${grpo_ckpt}" "${GRPO_DEVICES}" "${MASTER_PORT_GRPO}")"
  pid_reinforce="$(run_one_eval_bg "recogdrive_stage3_rl_reinforce_4nodes_v2" "${reinforce_ckpt}" "${REINFORCE_DEVICES}" "${MASTER_PORT_REINFORCE}")"

  echo "[RUNNING] GRPO_PID=${pid_grpo}, REINFORCE_PID=${pid_reinforce}"
  set +e
  wait "${pid_grpo}"
  rc_grpo=$?
  wait "${pid_reinforce}"
  rc_reinforce=$?
  set -e

  echo "[DONE] grpo_rc=${rc_grpo}, reinforce_rc=${rc_reinforce}"
  if [[ "${rc_grpo}" -ne 0 || "${rc_reinforce}" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
