#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NavTrain Trainval 数据集下载脚本 (鲁棒版)
# 功能：
# 1) 断点续传 (aria2c/wget)
# 2) 日志记录 (可断线后查看)
# 3) 已完成跳过 (.done 标记)
# 4) 避免写到 /root 等 home 目录
# 5) 可重跑：从中断处继续
# ==============================================================================

########################################
# 1. 核心配置
########################################
BASE_PATH="${BASE_PATH:-/data/dataset/navsim}"

# 避免在系统 home 下写缓存（例如 /root/.wget-hsts）
CACHE_HOME="${CACHE_HOME:-$BASE_PATH/.cache_home}"
mkdir -p "$CACHE_HOME"
export HOME="$CACHE_HOME"
export XDG_CACHE_HOME="$CACHE_HOME/.cache"
export XDG_CONFIG_HOME="$CACHE_HOME/.config"
export XDG_DATA_HOME="$CACHE_HOME/.local/share"

TMP_DOWNLOAD_DIR="${TMP_DOWNLOAD_DIR:-$BASE_PATH/tmp_download_navtrain}"
SENSOR_DATA_DIR="${SENSOR_DATA_DIR:-$BASE_PATH/trainval_sensor_blobs/trainval}"
LOG_DATA_DIR="${LOG_DATA_DIR:-$BASE_PATH/trainval_navsim_logs}"

LOG_DIR="${LOG_DIR:-$BASE_PATH/download_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$(basename "$0" .sh).log}"
STATE_DIR="${STATE_DIR:-$BASE_PATH/.download_state/$(basename "$0" .sh)}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# aria2c 限制：-x/--max-connection-per-server 取值范围 1-16
ARIA_X=${ARIA_X:-16}
ARIA_J=${ARIA_J:-2}
if (( ARIA_X < 1 )); then
    echo "警告: ARIA_X=$ARIA_X 非法，已重置为 1" >&2
    ARIA_X=1
elif (( ARIA_X > 16 )); then
    echo "警告: ARIA_X=$ARIA_X 超出 aria2c 支持范围(1-16)，已自动降为 16" >&2
    ARIA_X=16
fi
MAX_RETRY=${MAX_RETRY:-10}

# 1=解压并同步后删除 tgz；0=保留 tgz（便于手动校验/复用）
CLEAN_TGZ=${CLEAN_TGZ:-1}

########################################
# 2. 工具探测
########################################
A2C_CMD=""
if command -v aria2c >/dev/null 2>&1; then
    A2C_CMD="aria2c"
fi

WGET_CMD=""
if command -v wget >/dev/null 2>&1; then
    WGET_HSTS_FILE="$CACHE_HOME/wget-hsts"
    WGET_CMD="wget --hsts-file=$WGET_HSTS_FILE -c -O"
fi

if [[ -z "$A2C_CMD" && -z "$WGET_CMD" ]]; then
    echo "错误: 未找到 aria2c 或 wget，请先安装。" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "警告: 未找到 rsync，尝试使用 cp -r 代替 (可能会慢一点)..."
    SYNC_CMD="cp -r"
    SYNC_SUFFIX=""
else
    SYNC_CMD="rsync -a"
    SYNC_SUFFIX="/"
fi

mkdir -p "$BASE_PATH" "$TMP_DOWNLOAD_DIR" "$SENSOR_DATA_DIR" "$LOG_DATA_DIR" "$LOG_DIR" "$STATE_DIR"
cd "$BASE_PATH"

# 防止重复启动多个实例导致占带宽/互相踩文件
LOCK_FILE="$STATE_DIR/.lock"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "检测到已有实例在运行(锁: $LOCK_FILE)，本次直接退出以避免重复下载/解压。" >&2
        exit 0
    fi
else
    echo "警告: 未找到 flock，无法启用单实例锁（建议安装 util-linux）。" >&2
fi

if [[ "${DISABLE_LOG:-0}" != "1" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "=================================================="
echo "[启动] $(date -R)"
echo "BASE_PATH=$BASE_PATH"
echo "TMP_DOWNLOAD_DIR=$TMP_DOWNLOAD_DIR"
echo "SENSOR_DATA_DIR=$SENSOR_DATA_DIR"
echo "LOG_DATA_DIR=$LOG_DATA_DIR"
echo "STATE_DIR=$STATE_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "ARIA_X=$ARIA_X ARIA_J=$ARIA_J"
echo "=================================================="

download_one() {
    local url="$1"; shift
    local out="$1"; shift
    local dst="$TMP_DOWNLOAD_DIR/$out"
    local att=1

    # 完整文件已存在且可读取 tar 结构，则直接复用
    if [[ -f "$dst" && ! -f "$dst.aria2" ]]; then
        if tar -tzf "$dst" >/dev/null 2>&1; then
            echo ">> [复用] 已存在且校验通过: $out"
            return 0
        fi
    fi

    echo ">> [下载开始] $out"
    while :; do
        if [[ -n "$A2C_CMD" ]]; then
            if $A2C_CMD -c -x "$ARIA_X" -s "$ARIA_X" -j "$ARIA_J" -d "$TMP_DOWNLOAD_DIR" -o "$out" \
                 --connect-timeout=30 --timeout=600 --max-tries=5 "$url"; then
                return 0
            fi
        else
            if $WGET_CMD "$dst" "$url"; then
                return 0
            fi
        fi

        echo "!! [下载失败] $out (尝试 $att/$MAX_RETRY)" >&2
        att=$((att+1))
        if (( att > MAX_RETRY )); then
            echo "!! [放弃] 超过最大重试次数: $out" >&2
            return 2
        fi
        sleep 5
    done
}

extract_and_sync_dir() {
    local tgz_bn="$1"; shift
    local extracted_dir_name="$1"; shift
    local tgz_path="$TMP_DOWNLOAD_DIR/$tgz_bn"

    echo ">> [校验中] $tgz_bn"
    if ! tar -tzf "$tgz_path" >/dev/null 2>&1; then
        echo "!! [校验失败] 文件损坏，删除后重试: $tgz_bn" >&2
        rm -f "$tgz_path"
        return 3
    fi

    local EXTRACT_DIR
    EXTRACT_DIR=$(mktemp -d -p "$TMP_DOWNLOAD_DIR" "extract_XXXXXX")
    echo ">> [解压中] $tgz_bn -> $EXTRACT_DIR"
    tar -xzf "$tgz_path" -C "$EXTRACT_DIR"

    local SRC="$EXTRACT_DIR/$extracted_dir_name"
    if [[ ! -d "$SRC" ]]; then
        echo "!! [错误] 解压后未找到目录: $extracted_dir_name (在 $EXTRACT_DIR 下)" >&2
        rm -rf "$EXTRACT_DIR"
        return 4
    fi

    echo ">> [同步中] $extracted_dir_name -> $SENSOR_DATA_DIR"
    if [[ "$SYNC_CMD" == "rsync -a" ]]; then
        $SYNC_CMD "$SRC$SYNC_SUFFIX" "$SENSOR_DATA_DIR/"
    else
        cp -r "$SRC"/* "$SENSOR_DATA_DIR/"
    fi

    rm -rf "$EXTRACT_DIR"
    if [[ "$CLEAN_TGZ" == "1" ]]; then
        rm -f "$tgz_path"
    fi
}

########################################
# 3. Metadata (OpenScene trainval)
########################################
META_TGZ="openscene_metadata_trainval.tgz"
META_URL="$HF_ENDPOINT/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/$META_TGZ"

META_DONE_MARKER="$STATE_DIR/meta_datas.done"
if [[ -f "$META_DONE_MARKER" ]]; then
    echo "metadata 已存在完成标记，跳过。"
elif [[ -d "$LOG_DATA_DIR/meta_datas" ]] && [[ -n "$(ls -A "$LOG_DATA_DIR/meta_datas" 2>/dev/null)" ]]; then
    echo "metadata 目录已存在，补写完成标记并跳过。"
    touch "$META_DONE_MARKER"
else
    echo "=== 处理 Metadata ==="
    if download_one "$META_URL" "$META_TGZ"; then
        echo "解压 Metadata..."
        tar -xzf "$TMP_DOWNLOAD_DIR/$META_TGZ" -C "$TMP_DOWNLOAD_DIR"

        mkdir -p "$LOG_DATA_DIR/meta_datas"
        if [[ -d "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas" ]]; then
            if [[ "$SYNC_CMD" == "rsync -a" ]]; then
                $SYNC_CMD "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas$SYNC_SUFFIX" "$LOG_DATA_DIR/meta_datas/"
            else
                cp -r "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas"/* "$LOG_DATA_DIR/meta_datas/"
            fi
            touch "$META_DONE_MARKER"
        else
            echo "!! [错误] 解压后未找到 openscene-v1.1/meta_datas" >&2
            exit 1
        fi

        rm -rf "$TMP_DOWNLOAD_DIR/openscene-v1.1"
        if [[ "$CLEAN_TGZ" == "1" ]]; then rm -f "$TMP_DOWNLOAD_DIR/$META_TGZ"; fi
    else
        echo "Metadata 下载失败，程序退出。" >&2
        exit 1
    fi
fi

########################################
# 4. NavTrain (current/history split 1..4)
########################################
S3_BASE="${NAVTRAIN_S3_BASE:-https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim}"

progress_summary() {
    local done
    done=$(find "$STATE_DIR" -maxdepth 1 -type f -name 'navtrain_*.done' 2>/dev/null | wc -l || true)
    echo "[进度] navtrain done: ${done}/8"
}

progress_summary

for split in 1 2 3 4; do
    echo
    echo "=================================================="
    echo "NavTrain CURRENT split: $split"
    echo "=================================================="

    marker="$STATE_DIR/navtrain_current_${split}.done"
    if [[ -f "$marker" ]]; then
        echo ">> [跳过] navtrain_current_${split} 已完成"
        continue
    fi

    tgz="navtrain_current_${split}.tgz"
    url="$S3_BASE/$tgz"
    SUCCESS=0
    for retry in 1 2; do
        if download_one "$url" "$tgz"; then
            if extract_and_sync_dir "$tgz" "current_split_${split}"; then
                SUCCESS=1
                touch "$marker"
                break
            else
                echo "!! [警告] current split $split 解压/同步失败，准备重试..." >&2
            fi
        fi
    done
    if [[ $SUCCESS -ne 1 ]]; then
        echo "!! [失败] navtrain_current_${split} 处理失败" >&2
    fi
    progress_summary
done

for split in 1 2 3 4; do
    echo
    echo "=================================================="
    echo "NavTrain HISTORY split: $split"
    echo "=================================================="

    marker="$STATE_DIR/navtrain_history_${split}.done"
    if [[ -f "$marker" ]]; then
        echo ">> [跳过] navtrain_history_${split} 已完成"
        continue
    fi

    tgz="navtrain_history_${split}.tgz"
    url="$S3_BASE/$tgz"
    SUCCESS=0
    for retry in 1 2; do
        if download_one "$url" "$tgz"; then
            if extract_and_sync_dir "$tgz" "history_split_${split}"; then
                SUCCESS=1
                touch "$marker"
                break
            else
                echo "!! [警告] history split $split 解压/同步失败，准备重试..." >&2
            fi
        fi
    done
    if [[ $SUCCESS -ne 1 ]]; then
        echo "!! [失败] navtrain_history_${split} 处理失败" >&2
    fi
    progress_summary
done

echo
echo "所有任务结束。"
