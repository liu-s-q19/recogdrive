#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# OpenScene Test 数据集下载脚本 (鲁棒版)
# 改动说明：
# 1. 范围调整为 0-31
# 2. 目录调整为 test_sensor_blobs 和 test_navsim_logs
# 3. 链接调整为 test 版本
# ==============================================================================

########################################
# 1. 核心配置 (已改为 Test 设置)
########################################
# 设置基础路径，默认当前目录
BASE_PATH="${BASE_PATH:-$(pwd)}"

# 临时下载缓存目录
TMP_DOWNLOAD_DIR=${TMP_DOWNLOAD_DIR:-"$BASE_PATH/tmp_download_test"}

# 最终数据存放目录 (对应你简易脚本里的 mv 目标)
SENSOR_DATA_DIR=${SENSOR_DATA_DIR:-"$BASE_PATH/test_sensor_blobs"}
LOG_DATA_DIR=${LOG_DATA_DIR:-"$BASE_PATH/test_navsim_logs"}

# 分片范围: Test 集是 0 到 31
SPLIT_START=${SPLIT_START:-0}
SPLIT_END=${SPLIT_END:-31}

# 使用国内镜像加速
HF_ENDPOINT="https://hf-mirror.com"

# 下载参数配置
ARIA_X=${ARIA_X:-16}      # 16线程下载
MAX_RETRY=${MAX_RETRY:-10} # 单个文件最大重试10次

# 开关：解压后是否删除 .tgz 文件 (1=删除, 0=保留)
CLEAN_TGZ=${CLEAN_TGZ:-1}

# URL 定义 (修改为 Test 集链接)
META_TGZ="openscene_metadata_test.tgz"
META_URL="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/${META_TGZ}"

CAM_PREFIX="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera"
LIDAR_PREFIX="${HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar"

########################################
# 2. 环境检查与工具函数
########################################

# 优先检测 aria2c
A2C_CMD=""
if command -v aria2c >/dev/null 2>&1; then
  A2C_CMD="aria2c"
fi
# 降级使用 wget
WGET_CMD=""
if command -v wget >/dev/null 2>&1; then
  WGET_CMD="wget -c -O"
fi

if [[ -z "$A2C_CMD" && -z "$WGET_CMD" ]]; then
  echo "错误: 未找到 aria2c 或 wget，请先安装。" >&2
  exit 1
fi

# 检查 rsync (必须有，用于合并目录)
if ! command -v rsync >/dev/null 2>&1; then
    echo "警告: 未找到 rsync，尝试使用 cp -r 代替 (可能会慢一点)..."
    SYNC_CMD="cp -r"
    # 如果是 cp，源目录结尾不能带 /
    SYNC_SUFFIX=""
else
    SYNC_CMD="rsync -a"
    # rsync 需要源目录结尾带 / 来表示复制内容
    SYNC_SUFFIX="/"
fi

# 创建目录
mkdir -p "$TMP_DOWNLOAD_DIR" "$SENSOR_DATA_DIR" "$LOG_DATA_DIR"

# --- 函数: 下载单个文件 (带重试) ---
download_one() {
  local url="$1"; shift
  local out="$1"; shift
  local att=1

  echo ">> [下载开始] $out"
  
  while :; do
    if [[ -n "$A2C_CMD" ]]; then
      # aria2c: 多线程 + 断点续传
      if $A2C_CMD -c -x "$ARIA_X" -s "$ARIA_X" -j 4 -d "$TMP_DOWNLOAD_DIR" -o "$out" \
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
    return 3 
  fi

  # 2. 解压到临时目录
  local EXTRACT_DIR
  EXTRACT_DIR=$(mktemp -d -p "$TMP_DOWNLOAD_DIR" "extract_XXXXXX")
  echo ">> [解压中] $tgz_bn -> $EXTRACT_DIR"
  tar -xzf "$tgz_path" -C "$EXTRACT_DIR"

  # 3. 移动/同步数据
  local SRC_PATH=""
  # 自动寻找 sensor_blobs 目录
  # Test 集解压后通常也在 openscene-v1.1/sensor_blobs 下，里面可能有 test 子目录
  if [[ -d "$EXTRACT_DIR/openscene-v1.1/sensor_blobs" ]]; then
    SRC_PATH="$EXTRACT_DIR/openscene-v1.1/sensor_blobs"
  elif [[ -d "$EXTRACT_DIR/sensor_blobs" ]]; then
    SRC_PATH="$EXTRACT_DIR/sensor_blobs"
  else
    # 兜底
    SRC_PATH="$EXTRACT_DIR"
  fi

  echo ">> [同步中] 移动数据到 $SENSOR_DATA_DIR"
  # 注意：这里我们将 source 下的所有内容同步到目标
  # 如果 source 是 .../sensor_blobs，它里面应该包含 test 文件夹或者直接是 blobs
  # 为了匹配你的原始逻辑 (mv .../sensor_blobs test_sensor_blobs)，我们这里做合并
  
  if [[ "$SYNC_CMD" == "rsync -a" ]]; then
      $SYNC_CMD "$SRC_PATH$SYNC_SUFFIX" "$SENSOR_DATA_DIR/"
  else
      # cp 处理逻辑 (防止目录嵌套)
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
if [[ -d "$LOG_DATA_DIR/meta_datas" ]] || [[ -f "$LOG_DATA_DIR/meta_datas.json" ]]; then
    echo "metadata 似乎已存在，跳过。"
else
    echo "=== 处理 Metadata ==="
    if download_one "$META_URL" "$META_TGZ"; then
        echo "解压 Metadata..."
        tar -xzf "$TMP_DOWNLOAD_DIR/$META_TGZ" -C "$TMP_DOWNLOAD_DIR"
        
        # 目标: mv openscene-v1.1/meta_datas test_navsim_logs
        SRC_META="$TMP_DOWNLOAD_DIR/openscene-v1.1/meta_datas"
        
        if [[ -d "$SRC_META" ]]; then
            echo "同步 Metadata 到 $LOG_DATA_DIR"
            if [[ "$SYNC_CMD" == "rsync -a" ]]; then
                $SYNC_CMD "$SRC_META$SYNC_SUFFIX" "$LOG_DATA_DIR/"
            else
                cp -r "$SRC_META"/* "$LOG_DATA_DIR/"
            fi
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

for (( sp=$SPLIT_START; sp<=SPLIT_END; sp++ )); do
  echo
  echo "=================================================="
  echo "Processing Split: $sp / $SPLIT_END"
  echo "=================================================="

  # --- 处理 Camera ---
  cam_tgz="openscene_sensor_test_camera_${sp}.tgz"
  cam_url="$CAM_PREFIX/$cam_tgz"
  
  SUCCESS=0
  for retry in {1..2}; do
      if download_one "$cam_url" "$cam_tgz"; then
          if extract_and_sync_one "$cam_tgz"; then
              SUCCESS=1
              break
          else
              echo "!! [警告] Camera split $sp 校验失败，准备重新下载..."
          fi
      fi
  done
  if [[ $SUCCESS -eq 0 ]]; then FAIL_LIST+=("CAMERA_$sp"); fi

  # --- 处理 Lidar ---
  lidar_tgz="openscene_sensor_test_lidar_${sp}.tgz"
  lidar_url="$LIDAR_PREFIX/$lidar_tgz"
  
  SUCCESS=0
  for retry in {1..2}; do
      if download_one "$lidar_url" "$lidar_tgz"; then
          if extract_and_sync_one "$lidar_tgz"; then
              SUCCESS=1
              break
          else
              echo "!! [警告] Lidar split $sp 校验失败，准备重新下载..."
          fi
      fi
  done
  if [[ $SUCCESS -eq 0 ]]; then FAIL_LIST+=("LIDAR_$sp"); fi
  
done

echo
echo "所有任务结束。"
if (( ${#FAIL_LIST[@]} > 0 )); then
  echo "以下 Split 处理失败 (请检查网络或手动重试):"
  for item in "${FAIL_LIST[@]}"; do echo " - $item"; done
  exit 1
fi