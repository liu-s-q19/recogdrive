#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# OpenScene Trainval 数据集下载脚本 (鲁棒版)
# 功能：
# 1. 自动重试 (解决丢包/网络中断)
# 2. 自动校验 (下载完立即尝试读取 tar 结构，坏包自动重下)
# 3. 节省空间 (下载一个 -> 解压 -> 删包)
# ==============================================================================

########################################
# 1. 核心配置 (已改为 Trainval 设置)
########################################
# 设置基础路径：默认固定到 /data/dataset/navsim，避免在 home 下产生缓存
BASE_PATH="${BASE_PATH:-/data/dataset/navsim}"

# 避免 wget/aria2c/临时文件在系统 home 下写缓存（例如 /root/.wget-hsts）
CACHE_HOME="${CACHE_HOME:-$BASE_PATH/.cache_home}"
mkdir -p "$CACHE_HOME"
export HOME="$CACHE_HOME"
export XDG_CACHE_HOME="$CACHE_HOME/.cache"
export XDG_CONFIG_HOME="$CACHE_HOME/.config"
export XDG_DATA_HOME="$CACHE_HOME/.local/share"

# 临时下载缓存目录
TMP_DOWNLOAD_DIR=${TMP_DOWNLOAD_DIR:-"$BASE_PATH/tmp_download"}

# 最终数据存放目录 (对应你脚本里的 trainval_sensor_blobs 和 trainval_navsim_logs)
SENSOR_DATA_DIR=${SENSOR_DATA_DIR:-"$BASE_PATH/trainval_sensor_blobs"}
LOG_DATA_DIR=${LOG_DATA_DIR:-"$BASE_PATH/trainval_navsim_logs"}

# 分片范围: Trainval 集是 0 到 199
SPLIT_START=${SPLIT_START:-0}
SPLIT_END=${SPLIT_END:-199}

# 使用国内镜像加速 (如果服务器在海外，可把 hf-mirror.com 改回 huggingface.co)
HF_ENDPOINT="https://hf-mirror.com"
# HF_ENDPOINT="https://huggingface.co" 

# 下载参数配置
# aria2c 限制：-x/--max-connection-per-server 取值范围 1-16
ARIA_X=${ARIA_X:-16}
ARIA_J=${ARIA_J:-4}       # 同时并行下载任务数（不是单文件连接数）
if (( ARIA_X < 1 )); then
  echo "警告: ARIA_X=$ARIA_X 非法，已重置为 1" >&2
  ARIA_X=1
elif (( ARIA_X > 16 )); then
  echo "警告: ARIA_X=$ARIA_X 超出 aria2c 支持范围(1-16)，已自动降为 16" >&2
  ARIA_X=16
fi
MAX_RETRY=${MAX_RETRY:-10} # 单个文件最大重试10次

# 日志与状态
LOG_DIR=${LOG_DIR:-"$BASE_PATH/download_logs"}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/$(basename "$0" .sh).log"}
STATE_DIR=${STATE_DIR:-"$BASE_PATH/.download_state/$(basename "$0" .sh)"}

# 开关：解压后是否删除 .tgz 文件 (1=删除, 0=保留)
CLEAN_TGZ=${CLEAN_TGZ:-1}

# URL 定义
META_TGZ="openscene_metadata_trainval.tgz"
META_URL="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/${META_TGZ}"

CAM_PREFIX="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera"
LIDAR_PREFIX="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar"

########################################
# 2. 环境检查与工具函数
########################################

# 优先检测 aria2c (下载神器，强烈建议安装: apt install aria2 或 conda install aria2)
A2C_CMD=""
if command -v aria2c >/dev/null 2>&1; then
  A2C_CMD="aria2c"
fi
# 降级使用 wget
WGET_CMD=""
if command -v wget >/dev/null 2>&1; then
  WGET_HSTS_FILE="$CACHE_HOME/wget-hsts"
  WGET_CMD="wget --hsts-file=$WGET_HSTS_FILE -c -O" # -c 开启断点续传
fi

if [[ -z "$A2C_CMD" && -z "$WGET_CMD" ]]; then
  echo "错误: 未找到 aria2c 或 wget，请先安装。" >&2
  exit 1
fi

# 检查 rsync（推荐）。没有 rsync 时自动降级用 cp -r（可能更慢）
if ! command -v rsync >/dev/null 2>&1; then
  echo "警告: 未找到 rsync，尝试使用 cp -r 代替 (可能会慢一点)..."
  SYNC_CMD="cp -r"
  SYNC_SUFFIX=""
else
  SYNC_CMD="rsync -a"
  SYNC_SUFFIX="/"
fi

# 创建目录
mkdir -p "$BASE_PATH" "$TMP_DOWNLOAD_DIR" "$SENSOR_DATA_DIR" "$LOG_DATA_DIR" "$LOG_DIR" "$STATE_DIR"

# 强制在 BASE_PATH 下工作，避免把文件下载到其他目录（例如 ~）
cd "$BASE_PATH"

# 默认记录日志到文件，便于断线后查看
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

progress_summary() {
  local done_cam done_lidar
  done_cam=$(find "$STATE_DIR" -maxdepth 1 -type f -name 'camera_*.done' 2>/dev/null | wc -l || true)
  done_lidar=$(find "$STATE_DIR" -maxdepth 1 -type f -name 'lidar_*.done' 2>/dev/null | wc -l || true)
  echo "[进度] camera: ${done_cam}/$((SPLIT_END - SPLIT_START + 1)), lidar: ${done_lidar}/$((SPLIT_END - SPLIT_START + 1))"
}

# --- 函数: 下载单个文件 (带重试) ---
download_one() {
  local url="$1"; shift
  local out="$1"; shift
  local att=1

  # 如果完整文件已存在，尽量直接复用；如果存在 aria2 的控制文件，交给 aria2 续传
  local dst="$TMP_DOWNLOAD_DIR/$out"
  if [[ -f "$dst" && ! -f "$dst.aria2" ]]; then
    if tar -tzf "$dst" >/dev/null 2>&1; then
      echo ">> [复用] 已存在且校验通过: $out"
      return 0
    fi
  fi

  # 如果文件已存在且不想覆盖，可以在这里加判断，但为了鲁棒性，通常交给 aria2 处理断点续传
  echo ">> [下载开始] $out"
  
  while :; do
    if [[ -n "$A2C_CMD" ]]; then
      # aria2c: 多线程 + 断点续传
      if $A2C_CMD -c -x "$ARIA_X" -s "$ARIA_X" -j "$ARIA_J" -d "$TMP_DOWNLOAD_DIR" -o "$out" \
         --connect-timeout=30 --timeout=600 --max-tries=5 "$url"; then
        return 0
      fi
    elif [[ -n "$WGET_CMD" ]]; then
      # wget
      if $WGET_CMD "$TMP_DOWNLOAD_DIR/$out" "$url"; then
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

# --- 函数: 解压并同步 (带校验) ---
extract_and_sync_one() {
  local tgz_bn="$1"; shift
  local tgz_path="$TMP_DOWNLOAD_DIR/$tgz_bn"

  # 1. 完整性校验
  echo ">> [校验中] 正在检查包完整性: $tgz_bn"
  if ! tar -tzf "$tgz_path" >/dev/null 2>&1; then
    echo "!! [校验失败] 文件损坏，尝试删除并返回错误: $tgz_bn" >&2
    rm -f "$tgz_path"
    return 3 # 返回错误码，触发外层逻辑处理(比如重新下载)
  fi

  # 2. 解压到临时目录
  local EXTRACT_DIR
  EXTRACT_DIR=$(mktemp -d -p "$TMP_DOWNLOAD_DIR" "extract_XXXXXX")
  echo ">> [解压中] $tgz_bn -> $EXTRACT_DIR"
  tar -xzf "$tgz_path" -C "$EXTRACT_DIR"

  # 3. 移动/同步数据
  # 查找解压后的 sensor_blobs 路径 (通常在 openscene-v1.1/sensor_blobs/trainval 或类似路径)
  # 为了通用，我们查找包含 camera 或 lidar 的目录
  local SRC_PATH=""
  # 尝试定位数据根目录
  if [[ -d "$EXTRACT_DIR/openscene-v1.1/sensor_blobs/trainval" ]]; then
    SRC_PATH="$EXTRACT_DIR/openscene-v1.1/sensor_blobs/trainval"
  elif [[ -d "$EXTRACT_DIR/sensor_blobs/trainval" ]]; then
    SRC_PATH="$EXTRACT_DIR/sensor_blobs/trainval"
  else
    # 兜底: 如果目录结构变了，直接同步所有内容
    SRC_PATH="$EXTRACT_DIR"
  fi

  echo ">> [同步中] 移动数据到 $SENSOR_DATA_DIR"
  # 合并目录：优先 rsync，缺失则降级 cp
  if [[ "$SYNC_CMD" == "rsync -a" ]]; then
    $SYNC_CMD "$SRC_PATH$SYNC_SUFFIX" "$SENSOR_DATA_DIR/"
  else
    cp -r "$SRC_PATH"/* "$SENSOR_DATA_DIR/"
  fi

  # 4. 清理
  rm -rf "$EXTRACT_DIR"
  if [[ "$CLEAN_TGZ" == "1" ]]; then
    echo ">> [清理] 删除压缩包 $tgz_bn"
    rm -f "$tgz_path"
  fi
}

########################################
# 3. 主流程: 处理 Metadata
########################################
META_DONE_MARKER="$STATE_DIR/meta_datas.done"
if [[ -f "$META_DONE_MARKER" ]] || { [[ -d "$LOG_DATA_DIR/meta_datas" ]] && [[ -n "$(ls -A "$LOG_DATA_DIR/meta_datas" 2>/dev/null)" ]]; }; then
  echo "metadata 似乎已存在，跳过。"
else
    echo "=== 处理 Metadata ==="
    if download_one "$META_URL" "$META_TGZ"; then
        echo "解压 Metadata..."
        tar -xzf "$TMP_DOWNLOAD_DIR/$META_TGZ" -C "$TMP_DOWNLOAD_DIR"
        
        # 根据你脚本的逻辑：mv openscene-v1.1/meta_datas trainval_navsim_logs
        # 这里我们把 meta_datas 里的内容放进 LOG_DATA_DIR
        if [[ -d "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas" ]]; then
          echo "同步 Metadata 到 $LOG_DATA_DIR/meta_datas"
          mkdir -p "$LOG_DATA_DIR/meta_datas"
          if [[ "$SYNC_CMD" == "rsync -a" ]]; then
            $SYNC_CMD "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas$SYNC_SUFFIX" "$LOG_DATA_DIR/meta_datas/"
          else
            cp -r "$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas"/* "$LOG_DATA_DIR/meta_datas/"
          fi
          touch "$META_DONE_MARKER"
        fi
        
        rm -rf "$TMP_DOWNLOAD_DIR/openscene-v1.1"
        if [[ "$CLEAN_TGZ" == "1" ]]; then rm -f "$TMP_DOWNLOAD_DIR/$META_TGZ"; fi
    else
        echo "Metadata 下载失败，程序退出。"
        exit 1
    fi
fi

########################################
# 4. 主流程: 循环处理 Splits (Camera + Lidar)
########################################
FAIL_LIST=()

progress_summary

# 自动续跑：
# - 如果已经存在 .done 标记：从第一个未完成 split 开始
# - 否则如果 TMP_DOWNLOAD_DIR 里有中断的 tgz：从编号最大的那个 split 开始（更符合“从当前继续”的直觉）
AUTO_START_FROM_STATE=${AUTO_START_FROM_STATE:-1}
AUTO_PICKUP_FROM_TMP=${AUTO_PICKUP_FROM_TMP:-1}

SENSOR_PREFIX="openscene_sensor_trainval"

auto_find_first_incomplete_split() {
  local sp
  for (( sp=$SPLIT_START; sp<=SPLIT_END; sp++ )); do
    if [[ ! -f "$STATE_DIR/camera_${sp}.done" || ! -f "$STATE_DIR/lidar_${sp}.done" ]]; then
      echo "$sp"
      return 0
    fi
  done
  echo "$SPLIT_END"
}

auto_find_max_split_in_tmp() {
  local max=-1 f bn idx
  shopt -s nullglob
  for f in "$TMP_DOWNLOAD_DIR"/${SENSOR_PREFIX}_camera_*.tgz* "$TMP_DOWNLOAD_DIR"/${SENSOR_PREFIX}_lidar_*.tgz*; do
    bn=$(basename "$f")
    if [[ "$bn" =~ _camera_([0-9]+)\\.tgz(\\.aria2)?$ ]]; then
      idx=${BASH_REMATCH[1]}
    elif [[ "$bn" =~ _lidar_([0-9]+)\\.tgz(\\.aria2)?$ ]]; then
      idx=${BASH_REMATCH[1]}
    else
      continue
    fi
    if (( idx > max )); then max=$idx; fi
  done
  shopt -u nullglob
  echo "$max"
}

if [[ "$AUTO_START_FROM_STATE" == "1" ]]; then
  if compgen -G "$STATE_DIR/camera_*.done" >/dev/null 2>&1 || compgen -G "$STATE_DIR/lidar_*.done" >/dev/null 2>&1; then
    new_start=$(auto_find_first_incomplete_split)
    if (( new_start > SPLIT_START )); then
      echo "[自动续跑] 检测到已完成标记，将 SPLIT_START 从 $SPLIT_START 调整为 $new_start"
      SPLIT_START=$new_start
    fi
  elif [[ "$AUTO_PICKUP_FROM_TMP" == "1" ]]; then
    max_tmp=$(auto_find_max_split_in_tmp)
    if (( max_tmp >= 0 && max_tmp > SPLIT_START )); then
      echo "[自动续跑] 检测到中断下载文件，将 SPLIT_START 从 $SPLIT_START 调整为 $max_tmp"
      SPLIT_START=$max_tmp
    fi
  fi
fi

for (( sp=$SPLIT_START; sp<=SPLIT_END; sp++ )); do
  echo
  echo "=================================================="
  echo "Processing Split: $sp / $SPLIT_END"
  echo "=================================================="

  # --- 处理 Camera ---
  cam_done="$STATE_DIR/camera_${sp}.done"
  if [[ -f "$cam_done" ]]; then
    echo ">> [跳过] Camera split $sp 已完成"
  else
  cam_tgz="openscene_sensor_trainval_camera_${sp}.tgz"
  cam_url="$CAM_PREFIX/$cam_tgz"
  
  # 逻辑：下载 -> 如果成功 -> 解压。如果解压校验失败(返回3) -> 重新下载 -> 重新解压
  # 这里做一个简单的两轮尝试逻辑
  SUCCESS=0
  for retry in {1..2}; do
      if download_one "$cam_url" "$cam_tgz"; then
          if extract_and_sync_one "$cam_tgz"; then
              SUCCESS=1
          touch "$cam_done"
              break
          else
              echo "!! [警告] Camera split $sp 校验失败，准备重新下载..."
          fi
      fi
  done
  if [[ $SUCCESS -eq 0 ]]; then FAIL_LIST+=("CAMERA_$sp"); fi
    fi

  # --- 处理 Lidar ---
  lidar_done="$STATE_DIR/lidar_${sp}.done"
  if [[ -f "$lidar_done" ]]; then
    echo ">> [跳过] Lidar split $sp 已完成"
  else
  lidar_tgz="openscene_sensor_trainval_lidar_${sp}.tgz"
  lidar_url="$LIDAR_PREFIX/$lidar_tgz"
  
  SUCCESS=0
  for retry in {1..2}; do
      if download_one "$lidar_url" "$lidar_tgz"; then
          if extract_and_sync_one "$lidar_tgz"; then
              SUCCESS=1
          touch "$lidar_done"
              break
          else
              echo "!! [警告] Lidar split $sp 校验失败，准备重新下载..."
          fi
      fi
  done
  if [[ $SUCCESS -eq 0 ]]; then FAIL_LIST+=("LIDAR_$sp"); fi
    fi

    progress_summary
  
done

echo
echo "所有任务结束。"
if (( ${#FAIL_LIST[@]} > 0 )); then
  echo "以下 Split 处理失败 (请检查网络或手动重试):"
  for item in "${FAIL_LIST[@]}"; do echo " - $item"; done
  exit 1
fi