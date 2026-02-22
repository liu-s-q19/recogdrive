#!/usr/bin/env bash
set -euo pipefail

# 固定在 /data/dataset/navsim 下工作，避免落到 home/当前目录
BASE_PATH="${BASE_PATH:-/data/dataset/navsim}"
mkdir -p "$BASE_PATH"
cd "$BASE_PATH"

# 避免 wget/解压等工具在系统 home 下写缓存（例如 /root/.wget-hsts）
CACHE_HOME="${CACHE_HOME:-$BASE_PATH/.cache_home}"
mkdir -p "$CACHE_HOME"
export HOME="$CACHE_HOME"
export XDG_CACHE_HOME="$CACHE_HOME/.cache"
export XDG_CONFIG_HOME="$CACHE_HOME/.config"
export XDG_DATA_HOME="$CACHE_HOME/.local/share"

WGET_HSTS_FILE="$CACHE_HOME/wget-hsts"

URL="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
OUT="nuplan-maps-v1.1.zip"

echo ">> [下载开始] $OUT"
wget --hsts-file="$WGET_HSTS_FILE" -c -O "$OUT" \
	--retry-connrefused --waitretry=5 --read-timeout=30 --timeout=30 --tries=20 \
	"$URL"

echo ">> [完成] 已保存压缩包: $BASE_PATH/$OUT"
echo ">> 说明：按你的要求不解压、不删除，直接使用该 zip 文件。"
