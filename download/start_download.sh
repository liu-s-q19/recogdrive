#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 后台下载启动器
# 功能:
# 1. 使用 nohup 将下载脚本置于后台运行，关闭终端/断开连接后依然执行。
# 2. 将所有输出重定向到日志文件，方便随时查看进度。
# 3. 支持通过环境变量传递参数给下载脚本。
# ==============================================================================

# --- 用法 ---
#   bash start_download.sh tv   # Trainval
#   bash start_download.sh t    # Test
#   bash start_download.sh navtrain  # NavTrain (train)

MODE="${1:-t}"
case "$MODE" in
    tv)
        SCRIPT_TO_RUN="/data/liushiqi/recogdrive/download/download_tv.sh"
        ;;
    navtrain)
        SCRIPT_TO_RUN="/data/liushiqi/recogdrive/download/download_navtrain.sh"
        ;;
    t)
        SCRIPT_TO_RUN="/data/liushiqi/recogdrive/download/download_t.sh"
        ;;
    *)
        echo "用法: $0 [tv|t|navtrain]" >&2
        exit 2
        ;;
esac

BASE_PATH="${BASE_PATH:-/data/dataset/navsim}"
LOG_DIR="${LOG_DIR:-$BASE_PATH/download_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$(basename "$SCRIPT_TO_RUN" .sh).log}"

# --- 检查 ---
if [[ ! -f "$SCRIPT_TO_RUN" ]]; then
    echo "错误: 脚本不存在: $SCRIPT_TO_RUN" >&2
    exit 1
fi

# 检查是否已在运行 (简单的进程名匹配)
if ps -ef | grep -v grep | grep -q "$SCRIPT_TO_RUN"; then
    echo "警告: 检测到 '$SCRIPT_TO_RUN' 可能已在后台运行。"
    echo "请通过 'ps -ef | grep download' 或 'tail -f $LOG_FILE' 确认。"
    if [[ "${FORCE:-0}" == "1" ]]; then
        echo "FORCE=1: 继续启动新实例（可能会重复下载/占带宽）。"
    else
        read -p "是否仍然要强制启动一个新的实例? (y/N): " -r answer
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            echo "操作取消。"
            exit 0
        fi
    fi
fi

# --- 启动 ---
# 使用 nohup 启动脚本（脚本自身会 tee 到 LOG_FILE；这里把 stdout/stderr 丢弃避免重复写）
mkdir -p "$BASE_PATH" "$LOG_DIR"
nohup env BASE_PATH="$BASE_PATH" LOG_DIR="$LOG_DIR" LOG_FILE="$LOG_FILE" bash "$SCRIPT_TO_RUN" >/dev/null 2>&1 &

# 获取后台任务的进程ID (PID)
PID=$!

echo "=================================================="
echo "下载任务已在后台启动！"
echo "--------------------------------------------------"
echo "脚本: $SCRIPT_TO_RUN"
echo "PID: $PID"
echo "日志: $LOG_FILE"
echo "=================================================="
echo
echo "你可以通过以下命令实时查看进度:"
echo "tail -f $LOG_FILE"
echo
echo "如果需要停止任务，请运行:"
echo "kill $PID"
echo "=================================================="

