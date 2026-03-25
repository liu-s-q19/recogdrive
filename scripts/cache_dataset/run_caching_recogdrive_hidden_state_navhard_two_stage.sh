#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"

source /data/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-63669}"
VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="$RUNTIME_ROOT/exp"
export NAVSIM_OUTPUT_ROOT="$RUNTIME_ROOT/outputs"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/dataset/navsim}"
export TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_navhard_two_stage}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles}"

mkdir -p "$NAVSIM_EXP_ROOT" "$NAVSIM_OUTPUT_ROOT" "$TMPDIR" "$CACHE_PATH"

torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
    experiment_name=recogdrive_agent_cache_navhard_two_stage \
    train_test_split="$TRAIN_TEST_SPLIT" \
    cache_path="$CACHE_PATH" \
    cache_loader_mode="navsim_v2_scene_loader" \
    force_cache_computation=True \
    worker=sequential \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    original_sensor_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
    synthetic_scenes_path="$SYNTHETIC_SCENES_PATH" \
    agent=recogdrive_agent \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.vlm_type="internvl" \
    agent.vlm_path="$VLM_PATH" \
    agent.checkpoint_path=null \
    agent.grpo=False \
    agent.cache_loader_mode="navsim_v2_scene_loader" \
    hydra.job.chdir=False
