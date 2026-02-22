#!/usr/bin/env bash
set -euo pipefail

# 简版脚本：固定在 /data/dataset/navsim 下工作，避免落到 home/当前目录
BASE_PATH="${BASE_PATH:-/data/dataset/navsim}"
mkdir -p "$BASE_PATH"
cd "$BASE_PATH"

# 避免 wget 在系统 home 下写缓存（例如 /root/.wget-hsts）
CACHE_HOME="${CACHE_HOME:-$BASE_PATH/.cache_home}"
mkdir -p "$CACHE_HOME"
export HOME="$CACHE_HOME"
export XDG_CACHE_HOME="$CACHE_HOME/.cache"
export XDG_CONFIG_HOME="$CACHE_HOME/.config"
export XDG_DATA_HOME="$CACHE_HOME/.local/share"

wget https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz

for split in {0..199}; do
    wget https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${split}.tgz
    echo "Extracting file openscene_sensor_trainval_camera_${split}.tgz"
    tar -xzf openscene_sensor_trainval_camera_${split}.tgz
    rm openscene_sensor_trainval_camera_${split}.tgz
done

for split in {0..199}; do
    wget https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_${split}.tgz
    echo "Extracting file openscene_sensor_trainval_lidar_${split}.tgz"
    tar -xzf openscene_sensor_trainval_lidar_${split}.tgz
    rm openscene_sensor_trainval_lidar_${split}.tgz
done

mv openscene-v1.1/meta_datas trainval_navsim_logs
mv openscene-v1.1/sensor_blobs trainval_sensor_blobs
rm -r openscene-v1.1
