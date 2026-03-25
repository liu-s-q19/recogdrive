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

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
GPUS="${GPUS:-1}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$PROJECT_ROOT/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${RUNTIME_ROOT}/exp}"
export NAVSIM_OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/dataset/navsim}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
mkdir -p "${TMPDIR}"

MASTER_PORT="${MASTER_PORT:-63679}"
export MASTER_PORT

CHECKPOINT="${CHECKPOINT:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_ema_multinode_4nodes_8gpus/lightning_logs/version_1/checkpoints/epoch=99-step=8400-EMA.ckpt}"
VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH_EVAL="${CACHE_PATH_EVAL:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_navhard_two_stage}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage}"

temp_str="${CHECKPOINT#*outputs/}"
EXP_NAME="${EXP_NAME_OVERRIDE:-${temp_str%%/*}}"

OUTPUT_DIR="${OUTPUT_DIR:-$NAVSIM_OUTPUT_ROOT/${EXP_NAME}/eval_navhard_two_stage}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recogdrive_agent_eval_navhard_two_stage}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/eval_navhard_two_stage.log}"

mkdir -p "$OUTPUT_DIR"

torchrun \
    --nproc_per_node="${GPUS}" \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive_navhard.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path="$VLM_PATH" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.cache_loader_mode="navsim_v2_scene_loader" \
    cache_loader_mode="navsim_v2_scene_loader" \
    cache_path="$CACHE_PATH_EVAL" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    use_cache_without_dataset=false \
    agent.sampling_method="ddim" \
    worker=sequential \
    output_dir="$OUTPUT_DIR" \
    experiment_name="$EXPERIMENT_NAME" > "$LOG_FILE" 2>&1
