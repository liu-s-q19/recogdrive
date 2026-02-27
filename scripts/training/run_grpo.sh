#!/bin/bash
set -euo pipefail

source /data/miniconda/etc/profile.d/conda.sh
conda activate navsim

cd /data/liushiqi/recogdrive || exit 1

# ----------------- 1) 基础路径 -----------------
PROJECT_ROOT="/data/liushiqi/recogdrive"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/dataset/navsim/maps"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/dataset/navsim"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export TORCH_SHARING_STRATEGY=${TORCH_SHARING_STRATEGY:-file_descriptor}
export TMPDIR=${TMPDIR:-$NAVSIM_EXP_ROOT/tmp}
mkdir -p "$TMPDIR"

# ----------------- 2) 训练输入 -----------------
VLM_PATH="${VLM_PATH:-$PROJECT_ROOT/ckpt/ReCogDrive-VLM-8B}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_train}"

CHECKPOINT="${CHECKPOINT:-/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints}"
if [ -d "$CHECKPOINT" ]; then
    DEFAULT_CKPT_DIR="$CHECKPOINT"
else
    DEFAULT_CKPT_DIR="$(dirname "$CHECKPOINT")"
fi
DEFAULT_CKPT_EMA="$DEFAULT_CKPT_DIR/last-EMA.ckpt"
DEFAULT_CKPT_RAW="$DEFAULT_CKPT_DIR/last.ckpt"

if [ -f "$CHECKPOINT" ]; then
    :
elif [ -f "$DEFAULT_CKPT_EMA" ]; then
    CHECKPOINT="$DEFAULT_CKPT_EMA"
elif [ -f "$DEFAULT_CKPT_RAW" ]; then
    CHECKPOINT="$DEFAULT_CKPT_RAW"
else
    echo "[ERROR] No checkpoint file found."
    echo "        Input CHECKPOINT: $CHECKPOINT"
    echo "        Tried: $DEFAULT_CKPT_EMA"
    echo "        Tried: $DEFAULT_CKPT_RAW"
    exit 1
fi

# ----------------- 3) 分布式 -----------------
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

if [ -n "${PET_MASTER_ADDR:-}" ]; then
    MASTER_ADDR="$PET_MASTER_ADDR"
    MASTER_PORT="${PET_MASTER_PORT:-29500}"
    NODE_RANK=0
elif [ -n "${MLP_WORKER_0_HOST:-}" ]; then
    MASTER_ADDR="$MLP_WORKER_0_HOST"
    MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"
    NODE_RANK=0
else
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT="29500"
    NODE_RANK=0
fi

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export MASTER_ADDR
export MASTER_PORT

# ----------------- 4) 算法参数 -----------------
RL_ALGO="${RL_ALGO:-grpo_clip}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/recogdrive_grpo}"

GRPO_GAMMA="${GRPO_GAMMA:-0.6}"
GRPO_CLIP_LOW="${GRPO_CLIP_LOW:-0.00}"
GRPO_CLIP_HIGH="${GRPO_CLIP_HIGH:-1.00}"
GRPO_RANDN_CLIP="${GRPO_RANDN_CLIP:-5.0}"
GRPO_DENOISED_CLIP="${GRPO_DENOISED_CLIP:-1.0}"
GRPO_MIN_SAMPLING_STD="${GRPO_MIN_SAMPLING_STD:-0.04}"
GRPO_MIN_LOGPROB_STD="${GRPO_MIN_LOGPROB_STD:-0.1}"

SCORE_PROGRESS="${SCORE_PROGRESS:-10.0}"
SCORE_TTC="${SCORE_TTC:-5.0}"
SCORE_COMFORT="${SCORE_COMFORT:-2.0}"

GRPO_CLIP_EPS="${GRPO_CLIP_EPS:-0.2}"
GRPO_PPO_EPOCHS="${GRPO_PPO_EPOCHS:-2}"
GRPO_MINI_BATCH_SIZE="${GRPO_MINI_BATCH_SIZE:-16}"
GRPO_MAX_GRAD_NORM="${GRPO_MAX_GRAD_NORM:-1.0}"
GRPO_TARGET_KL="${GRPO_TARGET_KL:-0.03}"
GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-8}"
GRPO_BC_COEFF="${GRPO_BC_COEFF:-0.1}"
GRPO_USE_BC_LOSS="${GRPO_USE_BC_LOSS:-true}"

MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ----------------- 5) 持续监控策略 -----------------
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train_grpo_rank${NODE_RANK}_${RUN_TAG}.log"
STATUS_FILE="$OUTPUT_DIR/monitor_status_${RUN_TAG}.log"

MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-30}"
STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-1800}"
RESTART_BACKOFF_SEC="${RESTART_BACKOFF_SEC:-20}"
MAX_RESTARTS="${MAX_RESTARTS:--1}" # -1 表示无限重启

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "==================================================" | tee -a "$STATUS_FILE"
echo "🚀 GRPO Continuous Runner" | tee -a "$STATUS_FILE"
echo "Output Dir : $OUTPUT_DIR" | tee -a "$STATUS_FILE"
echo "Log File   : $LOG_FILE" | tee -a "$STATUS_FILE"
echo "Master     : $MASTER_ADDR:$MASTER_PORT" | tee -a "$STATUS_FILE"
echo "GPUs       : $GPUS_PER_NODE (single-node)" | tee -a "$STATUS_FILE"
echo "Algo       : $RL_ALGO" | tee -a "$STATUS_FILE"
echo "==================================================" | tee -a "$STATUS_FILE"

build_train_cmd() {
    cat <<EOF
torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE \
  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.vlm_path=$VLM_PATH \
  agent.cam_type='single' \
  agent.grpo=True \
  +agent.rl_algo_type=$RL_ALGO \
  +agent.grpo_cfg.gamma_denoising=$GRPO_GAMMA \
  +agent.grpo_cfg.clip_advantage_lower_quantile=$GRPO_CLIP_LOW \
  +agent.grpo_cfg.clip_advantage_upper_quantile=$GRPO_CLIP_HIGH \
  +agent.grpo_cfg.randn_clip_value=$GRPO_RANDN_CLIP \
  +agent.grpo_cfg.denoised_clip_value=$GRPO_DENOISED_CLIP \
  +agent.grpo_cfg.min_sampling_denoising_std=$GRPO_MIN_SAMPLING_STD \
  +agent.grpo_cfg.min_logprob_denoising_std=$GRPO_MIN_LOGPROB_STD \
  +agent.grpo_cfg.clip_epsilon=$GRPO_CLIP_EPS \
  +agent.grpo_cfg.ppo_epochs=$GRPO_PPO_EPOCHS \
  +agent.grpo_cfg.mini_batch_size=$GRPO_MINI_BATCH_SIZE \
  +agent.grpo_cfg.max_grad_norm=$GRPO_MAX_GRAD_NORM \
  +agent.grpo_cfg.target_kl=$GRPO_TARGET_KL \
  +agent.grpo_cfg.sample_time=$GRPO_SAMPLE_TIME \
  +agent.grpo_cfg.bc_coeff=$GRPO_BC_COEFF \
  +agent.grpo_cfg.use_bc_loss=$GRPO_USE_BC_LOSS \
  +agent.grpo_cfg.scorer_config.progress_weight=$SCORE_PROGRESS \
  +agent.grpo_cfg.scorer_config.ttc_weight=$SCORE_TTC \
  +agent.grpo_cfg.scorer_config.comfortable_weight=$SCORE_COMFORT \
  agent.cache_hidden_state=True \
  agent.vlm_type='internvl' \
  agent.checkpoint_path=$CHECKPOINT \
  agent.dit_type='small' \
  agent.sampling_method='ddim' \
  agent.metric_cache_path=$METRIC_CACHE_PATH \
  agent.reference_policy_checkpoint=$CHECKPOINT \
  trainer.params.max_epochs=$MAX_EPOCHS \
  dataloader.params.batch_size=$BATCH_SIZE \
  trainer.params.num_nodes=$NNODES \
  trainer.params.devices=$GPUS_PER_NODE \
  experiment_name=training_internvl_agent_dit_rl \
  train_test_split=$TRAIN_TEST_SPLIT \
  cache_path=$CACHE_PATH \
  output_dir=$OUTPUT_DIR \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  worker=sequential \
  $EXTRA_ARGS
EOF
}

start_once() {
    local train_cmd
    train_cmd="$(build_train_cmd)"
    echo "[$(date '+%F %T')] [RUN] launching training..." | tee -a "$STATUS_FILE"
    echo "[$(date '+%F %T')] [CMD] $train_cmd" >> "$STATUS_FILE"
    nohup bash -lc "$train_cmd" >> "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "[$(date '+%F %T')] [RUN] pid=$TRAIN_PID" | tee -a "$STATUS_FILE"
}

monitor_once() {
    local restarts=0
    while true; do
        start_once
        sleep 8

        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "[$(date '+%F %T')] [ERROR] 训练进程未成功拉起，8秒内已退出。" | tee -a "$STATUS_FILE"
            restarts=$((restarts + 1))
        else
            while kill -0 "$TRAIN_PID" 2>/dev/null; do
                local now
                now=$(date +%s)
                local mtime=0
                if [ -f "$LOG_FILE" ]; then
                    mtime=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
                fi

                local silence=$((now - mtime))
                local epoch_info
                epoch_info=$(grep -o "Epoch [0-9]\+" "$LOG_FILE" 2>/dev/null | tail -n 1 || true)
                local metric_info
                metric_info=$(grep -E "train/loss_step=|train/reward_step=" "$LOG_FILE" 2>/dev/null | tail -n 1 || true)

                echo "[$(date '+%F %T')] [MON] pid=$TRAIN_PID ${epoch_info:-Epoch N/A} silence=${silence}s" | tee -a "$STATUS_FILE"
                if [ -n "$metric_info" ]; then
                    echo "[$(date '+%F %T')] [MON] metric: $metric_info" | tee -a "$STATUS_FILE"
                fi

                if [ "$silence" -gt "$STALL_TIMEOUT_SEC" ]; then
                    echo "[$(date '+%F %T')] [WARN] 日志超过 ${STALL_TIMEOUT_SEC}s 无更新，判定卡住，杀进程并重启。" | tee -a "$STATUS_FILE"
                    pkill -P "$TRAIN_PID" || true
                    kill -9 "$TRAIN_PID" || true
                    break
                fi
                sleep "$MONITOR_INTERVAL_SEC"
            done

            if wait "$TRAIN_PID"; then
                EXIT_CODE=0
            else
                EXIT_CODE=$?
            fi
            if [ "$EXIT_CODE" -eq 0 ]; then
                echo "[$(date '+%F %T')] [OK] 训练正常结束。" | tee -a "$STATUS_FILE"
                return 0
            fi
            echo "[$(date '+%F %T')] [FAIL] 训练异常退出，code=$EXIT_CODE" | tee -a "$STATUS_FILE"
            restarts=$((restarts + 1))
        fi

        if [ "$MAX_RESTARTS" -ge 0 ] && [ "$restarts" -gt "$MAX_RESTARTS" ]; then
            echo "[$(date '+%F %T')] [STOP] 超过最大重启次数($MAX_RESTARTS)，停止。" | tee -a "$STATUS_FILE"
            return 1
        fi
        echo "[$(date '+%F %T')] [RESTART] 第${restarts}次重启，${RESTART_BACKOFF_SEC}s 后重试。" | tee -a "$STATUS_FILE"
        sleep "$RESTART_BACKOFF_SEC"
    done
}

monitor_once
