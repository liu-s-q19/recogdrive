# 加载 conda 配置
source /data/miniconda/etc/profile.d/conda.sh
# 激活你的虚拟环境
conda activate navsim
# 切换到代码根目录 (非常重要，否则 python 找不到模块)
cd /data/liushiqi/recogdrive

set -x

# ----------------- 1. 基础环境与路径配置 -----------------
TRAIN_TEST_SPLIT=navtest  # 评估集 Split

# 你的项目根目录
PROJECT_ROOT="/data/liushiqi/recogdrive"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/dataset/navsim"

# ----------------- 2. 显卡配置 (单机8卡) -----------------
# 评估通常不需要多机，单机 8 张 足够
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# 这里假设你使用一台机器上的 8 张卡进行并行评估
GPUS=8
echo "GPUS: ${GPUS}"
# export CUDA_LAUNCH_BLOCKING=1

# ----------------- 3. 关键模型路径 (请务必检查!) -----------------

# [A] 你的训练产物 (Checkpoint)
# 这是你刚刚训练完的模型。通常在 checkpoints 文件夹里会有 'last.ckpt' 或者 'epoch=xx.ckpt'
# 如果你找不到这个文件，请去文件夹里确认一下具体名字！
CHECKPOINT="/data/liushiqi/recogdrive/outputs/recogdrive_stage3_rl_grpo_4nodes_32gpus/lightning_logs/version_0/checkpoints/epoch=9-step=26600.ckpt"
CHECKPOINT_HYDRA="${CHECKPOINT//=/\\=}"

# [B] VLM 权重 (第一阶段产物，保持不变)
VLM_PATH="$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B"

# [C] 评估集缓存路径 (来自 run_caching_..._eval.sh)
# 【注意】这个路径必须存在！如果你之前没跑通 eval 缓存脚本，这里会报错。
CACHE_PATH_EVAL="$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_test"

# ----------------- 自动提取 EXP_NAME -----------------
# 1. ${CHECKPOINT#*outputs/} : 从左边开始删除，直到找到 'outputs/' 为止
#    结果变成: rl_A_clamp5/lightning_logs/...
temp_str="${CHECKPOINT#*outputs/}"

# 2. ${temp_str%%/*} : 从右边开始删除，保留第一个 '/' 左边的内容
#    结果变成: rl_A_clamp5
EXP_NAME="${temp_str%%/*}"

echo "Detected Experiment Name: ${EXP_NAME}"
# ----------------------------------------------------

# 现在可以使用提取出来的 EXP_NAME 了
OUTPUT_DIR="$PROJECT_ROOT/outputs/${EXP_NAME}/eval"

# ----------------- 4. 启动评估命令 -----------------
torchrun \
    --nproc_per_node=${GPUS} \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="$CHECKPOINT_HYDRA" \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    cache_path=$CACHE_PATH_EVAL \
    use_cache_without_dataset=True \
    agent.sampling_method="ddim" \
    worker=sequential \
    output_dir=$OUTPUT_DIR \
    experiment_name=recogdrive_agent_eval_shiqi > eval_shiqi.log 2>&1

