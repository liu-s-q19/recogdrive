set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# export NCCL_IB_DISABLE=1

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1


CHECKPOINT="/path/to/recogdrive.ckpt"


# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and run dataset caching
# cache_hidden_state:
# In IL/RL training, we save the VLM’s last_hidden_state together with historical trajectories, instructions, and ego status
# to accelerate training by avoiding repeated VLM forward passes. You can refer to navsim\agents\recogdrive\recogdrive_features.py.
# We are also exploring joint training of VLM and DiT, as VLM forward is the main bottleneck that slows training.
# During evaluation, we provide two modes: with or without caching the hidden state.
# For GPUs like 3090/4090, set cache_hidden_state=False to disable caching,
# allowing VLM and DiT to perform joint inference during evaluation.


torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/path/to/owl10/ReCogDrive-VLM-8B' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    cache_path="/path/to/exp/recogdrive_agent_cache_dir_train_test_8b" \
    use_cache_without_dataset=True \
    agent.sampling_method="ddim" \
    experiment_name=recogdrive_agent_eval > eval_8b.txt 2>&1

