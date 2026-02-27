#!/bin/bash
set -euo pipefail

# 增量回填训练集 metric cache（仅补缺失 token）
# 用法：
#   bash scripts/cache_dataset/run_metric_caching_backfill_missing_train.sh
# 可选环境变量：
#   BATCH_SIZE=128 MAX_WORKERS=32 PROJECT_ROOT=/data/liushiqi/recogdrive

source /data/miniconda/etc/profile.d/conda.sh
conda activate navsim

PROJECT_ROOT="${PROJECT_ROOT:-/data/liushiqi/recogdrive}"
cd "$PROJECT_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$PROJECT_ROOT/dataset/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/dataset/navsim}"

TRAIN_CACHE_PATH="${TRAIN_CACHE_PATH:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_train}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_WORKERS="${MAX_WORKERS:-32}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/metric_cache_backfill_navtrain}"
mkdir -p "$OUT_DIR"
MISSING_FILE="$OUT_DIR/missing_tokens.txt"
SUMMARY_FILE="$OUT_DIR/missing_summary.txt"

export TRAIN_CACHE_PATH METRIC_CACHE_PATH MISSING_FILE SUMMARY_FILE

python - <<'PY'
from pathlib import Path
import os
import subprocess
import re

train_cache = Path(os.environ["TRAIN_CACHE_PATH"])
metric_cache = Path(os.environ["METRIC_CACHE_PATH"])
out_file = Path(os.environ["MISSING_FILE"])
summary_file = Path(os.environ["SUMMARY_FILE"])


def load_tokens_from_cache(cache_root: Path):
    cmd = [
        "find",
        str(cache_root),
        "-type",
        "d",
        "-regextype",
        "posix-extended",
        "-regex",
        r".*/[0-9a-f]{16}$",
    ]
    out = subprocess.check_output(cmd, text=True)
    token_pattern = re.compile(r"[0-9a-f]{16}$")
    tokens = set()
    for line in out.splitlines():
        p = Path(line.strip())
        name = p.name
        if token_pattern.fullmatch(name):
            tokens.add(name)
    return tokens

train_tokens = load_tokens_from_cache(train_cache)
metric_tokens = load_tokens_from_cache(metric_cache)

missing = sorted(train_tokens - metric_tokens)
coverage = (len(metric_tokens & train_tokens) / len(train_tokens) * 100.0) if train_tokens else 0.0

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text("\n".join(missing) + ("\n" if missing else ""))

summary = (
    f"TRAIN_TOKENS={len(train_tokens)}\n"
    f"METRIC_TOKENS={len(metric_tokens)}\n"
    f"MISSING_TOKENS={len(missing)}\n"
    f"COVERAGE_PCT={coverage:.4f}\n"
)
summary_file.write_text(summary)
print(summary, end="")
PY

if [[ ! -s "$MISSING_FILE" ]]; then
  echo "No missing token found. Nothing to backfill."
  exit 0
fi

TOTAL=$(wc -l < "$MISSING_FILE" | tr -d ' ')
echo "Found $TOTAL missing tokens. Start backfill..."

# 清理历史分片，避免 rerun 时误处理旧批次
find "$OUT_DIR" -maxdepth 1 -name 'chunk_*.tokens' -delete

BATCH_ID=0
while IFS= read -r -d '' chunk; do
  BATCH_ID=$((BATCH_ID + 1))
    TOKEN_CNT=$(wc -l < "$chunk" | tr -d ' ')
  TOKENS_CSV=$(tr '\n' ',' < "$chunk" | sed 's/,$//')
  TOKENS_ARG="[$TOKENS_CSV]"

    echo "[Backfill] batch=${BATCH_ID} tokens=${TOKEN_CNT}"
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    cache.cache_path="$METRIC_CACHE_PATH" \
    worker=single_machine_thread_pool \
    worker.max_workers="$MAX_WORKERS" \
    worker.use_process_pool=True \
    "train_test_split.scene_filter.tokens=$TOKENS_ARG" \
    +experiment_name=metric_caching_backfill_navtrain

done < <(split -l "$BATCH_SIZE" -d -a 4 --additional-suffix=.tokens "$MISSING_FILE" "$OUT_DIR/chunk_" && find "$OUT_DIR" -maxdepth 1 -name 'chunk_*.tokens' -print0 | sort -z)

# 再次验证覆盖
python - <<'PY'
from pathlib import Path
import os
import subprocess
import re

train_cache = Path(os.environ["TRAIN_CACHE_PATH"])
metric_cache = Path(os.environ["METRIC_CACHE_PATH"])


def load_tokens_from_cache(cache_root: Path):
    cmd = [
        "find",
        str(cache_root),
        "-type",
        "d",
        "-regextype",
        "posix-extended",
        "-regex",
        r".*/[0-9a-f]{16}$",
    ]
    out = subprocess.check_output(cmd, text=True)
    token_pattern = re.compile(r"[0-9a-f]{16}$")
    tokens = set()
    for line in out.splitlines():
        p = Path(line.strip())
        name = p.name
        if token_pattern.fullmatch(name):
            tokens.add(name)
    return tokens

train_tokens = load_tokens_from_cache(train_cache)
metric_tokens = load_tokens_from_cache(metric_cache)
missing = sorted(train_tokens - metric_tokens)
coverage = (len(metric_tokens & train_tokens) / len(train_tokens) * 100.0) if train_tokens else 0.0
print(f"[PostCheck] TRAIN_TOKENS={len(train_tokens)} METRIC_TOKENS={len(metric_tokens)} MISSING_TOKENS={len(missing)} COVERAGE_PCT={coverage:.4f}")
PY

echo "Backfill finished. Logs and artifacts are in: $OUT_DIR"
