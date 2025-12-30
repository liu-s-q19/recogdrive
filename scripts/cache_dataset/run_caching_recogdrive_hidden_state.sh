TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive_rl/data/navsim/maps"
export NAVSIM_EXP_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive_rl/cache"
export NAVSIM_DEVKIT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive_rl"
export OPENSCENE_DATA_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive_rl/data/navsim"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"

torchrun \
    --nproc_per_node=2 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/ckpt/ReCogDrive-VLM-8B" \
    cache_path=$CACHE_PATH \
    worker=sequential > caching_dataset.txt 2>&1