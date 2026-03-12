#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qianxu/TensorRT-Model-Optimizer}"
ENV_NAME="${ENV_NAME:-modelopt}"
FULL_PRECISION_MODEL_DIR="${FULL_PRECISION_MODEL_DIR:-$REPO_ROOT/Qwen3-4B}"
TEMPLATE_NVFP4_DIR="${TEMPLATE_NVFP4_DIR:-$REPO_ROOT/Qwen3-4B-NVFP4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Qwen3-4B-NVFP4-compress10}"
PRESET="${PRESET:-compress10}"
DEVICE="${DEVICE:-cpu}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1GB}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/export_nvfp4_${PRESET}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

cat <<EOF
REPO_ROOT=$REPO_ROOT
FULL_PRECISION_MODEL_DIR=$FULL_PRECISION_MODEL_DIR
TEMPLATE_NVFP4_DIR=$TEMPLATE_NVFP4_DIR
OUTPUT_DIR=$OUTPUT_DIR
PRESET=$PRESET
DEVICE=$DEVICE
MAX_SHARD_SIZE=$MAX_SHARD_SIZE
LOG_FILE=$LOG_FILE
EOF

export PYTHONPATH="$REPO_ROOT"

conda run --no-capture-output -n "$ENV_NAME" python -u \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/export_double_scale_repo_mse_sweep.py" \
  --full-precision-model-dir "$FULL_PRECISION_MODEL_DIR" \
  --template-nvfp4-dir "$TEMPLATE_NVFP4_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  2>&1 | tee "$LOG_FILE"
