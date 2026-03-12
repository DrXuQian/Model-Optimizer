#!/usr/bin/env bash
set -euo pipefail

HOME="${HOME:-/sim/eec/shared/junfu.qx}"
MODEL_DIR="${MODEL_DIR:-/sim/eec/shared/models/Qwen}"
REPO_ROOT="${REPO_ROOT:-$HOME/Model-Optimizer}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_PRECISION_MODEL_DIR="${FULL_PRECISION_MODEL_DIR:-$MODEL_DIR/Qwen3-4B}"
TEMPLATE_NVFP4_DIR="${TEMPLATE_NVFP4_DIR:-$MODEL_DIR/Qwen3-4B-NVFP4}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Qwen3-4B-NVFP4-compress10}"
PRESET="${PRESET:-compress10}"
DEVICE="${DEVICE:-cuda}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1GB}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/export_nvfp4_${PRESET}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

cat <<EOF
HOME=$HOME
MODEL_DIR=$MODEL_DIR
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

"$PYTHON_BIN" -u \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/export_double_scale_repo_mse_sweep.py" \
  --full-precision-model-dir "$FULL_PRECISION_MODEL_DIR" \
  --template-nvfp4-dir "$TEMPLATE_NVFP4_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  2>&1 | tee "$LOG_FILE"
