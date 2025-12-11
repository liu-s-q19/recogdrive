#!/usr/bin/env bash
set -e  # 遇到错误立即停止

# ==============================================================================
# Hugging Face 模型下载脚本 (国内优化版)
# 功能：
# 1. 自动配置 hf-mirror 国内镜像 (解决连接超时)
# 2. 自动安装/检查依赖工具
# 3. 支持断点续传 (下载中断后重跑脚本，会从断点继续)
# 4. 下载实体文件 (不使用软链接，方便查看和移动)
# ==============================================================================

########################################
# 1. 核心配置
########################################

# 模型 ID (Hugging Face 上的名字)
MODEL_ID="owl10/ReCogDrive-VLM-8B"

# 存放路径 (建议使用绝对路径，这里默认放在当前目录下的 ReCogDrive-VLM-8B 文件夹)
BASE_PATH="${BASE_PATH:-$(pwd)}"
OUTPUT_DIR="${BASE_PATH}/ReCogDrive-VLM-8B"

# --- 关键：国内镜像配置 ---
# 只有设置了这个，huggingface-cli 才会走国内镜像
export HF_ENDPOINT="https://hf-mirror.com"

########################################
# 2. 环境检查与准备
########################################

echo "=================================================="
echo "正在检查运行环境..."
echo "=================================================="

# 检查 python 是否存在
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 conda 环境。"
    exit 1
fi

# 检查 huggingface_hub 工具是否存在，不存在则自动安装
if ! command -v huggingface-cli &> /dev/null; then
    echo ">> 未找到 huggingface-cli，正在尝试使用 pip 安装..."
    # 使用清华源加速安装工具本身
    pip install -U huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    if ! command -v huggingface-cli &> /dev/null; then
        echo "错误: huggingface-cli 安装失败，请手动运行: pip install huggingface_hub"
        exit 1
    fi
    echo ">> huggingface-cli 安装成功。"
else
    echo ">> 检测到 huggingface-cli 已安装，准备就绪。"
fi

########################################
# 3. 执行下载
########################################

echo "=================================================="
echo "开始下载模型: $MODEL_ID"
echo "镜像地址: $HF_ENDPOINT"
echo "保存位置: $OUTPUT_DIR"
echo "=================================================="

# 解释参数：
# download: 下载命令
# --resume-download: 显式开启断点续传 (新版默认开启，加了更保险)
# --local-dir: 指定下载到哪个文件夹
# --local-dir-use-symlinks False: 核心参数！下载真实文件而不是缓存链接，方便你拷贝
# --token: 如果是私有模型需要加这个参数，公开模型不需要

huggingface-cli download \
    --resume-download \
    "$MODEL_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False \
    --exclude "*.msgpack" "*.h5"  # (可选) 排除一些不用下载的非 PyTorch 格式文件，节省时间

echo ""
echo "=================================================="
echo "下载任务完成！"
echo "文件已保存在: $OUTPUT_DIR"
echo "=================================================="