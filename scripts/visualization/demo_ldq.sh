#!/bin/bash
set -x

# ----------------- 1. 基础路径配置 -----------------
# 你的项目根目录
PROJECT_ROOT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive"

# 导出环境变量 (指向你的真实数据路径)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/data/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/data/navsim"

# ----------------- 2. 显卡与通信配置 (适配 2卡 H20) -----------------
export NCCL_IB_DISABLE=1          # 禁用 IB，防止单机环境通信卡死
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}" # 确保 Python 能找到 navsim 包

# 单机配置
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29600                 # 换个端口，防止和训练任务冲突
GPUS=2                            # 你有 2 张卡，开启 2 个进程并行生成视频

echo "Generating Videos on ${GPUS} GPUs..."

# ----------------- 3. 模型与数据路径 -----------------
# [关键] 指向你 Stage 3 RL 训练出的最佳模型 (last.ckpt)
# 请确认这个路径存在！
CHECKPOINT="/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/exp/recogdrive_stage3_rl_training_16gpus_bs8/lightning_logs/version_0/checkpoints/epoch=9-step=6650.ckpt"

# Stage 1 VLM 权重
VLM_PATH="$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B"

# Navtest 评估集缓存 (可视化需要读取这些特征)
CACHE_PATH="$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_test"

# ----------------- 4. 启动命令 -----------------
# 注意：plt_all_log.py 需要和评估脚本一样的参数来加载 Agent
# 我们设置 train_test_split=navtest 来可视化测试集表现

# 【新增】导出变量供 Python 脚本读取
export CHECKPOINT_PATH=$CHECKPOINT
export VLM_PATH=$VLM_PATH

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/plt_all_log.py \
    train_test_split=navtest \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.sampling_method="ddim" \
    cache_path=$CACHE_PATH \
    use_cache_without_dataset=True \
    worker=sequential \
    experiment_name=visualization_video_demo