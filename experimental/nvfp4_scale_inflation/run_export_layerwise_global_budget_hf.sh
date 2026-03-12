#!/usr/bin/env bash
set -euo pipefail

HOME="/sim/eec/shared/junfu.qx"
MODEL_DIR="${MODEL_DIR:-/sim/eec/shared/models/Qwen}"
REPO_ROOT="${REPO_ROOT:-$HOME/Model-Optimizer}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_PRECISION_MODEL_DIR="${FULL_PRECISION_MODEL_DIR:-$MODEL_DIR/Qwen3-4B}"
TEMPLATE_NVFP4_DIR="${TEMPLATE_NVFP4_DIR:-$MODEL_DIR/Qwen3-4B-NVFP4}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Qwen3-4B-NVFP4-layerwise-global-budget}"
PRESET="${PRESET:-compress10}"
LAYERWISE_PROFILE="${LAYERWISE_PROFILE:-llm_sensitivity_v1}"
LAYERWISE_RULES_JSON="${LAYERWISE_RULES_JSON:-}"
DEVICE="${DEVICE:-cuda}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1GB}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/export_layerwise_global_budget_${PRESET}_${LAYERWISE_PROFILE}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

cat <<EOF
HOME=$HOME
MODEL_DIR=$MODEL_DIR
REPO_ROOT=$REPO_ROOT
FULL_PRECISION_MODEL_DIR=$FULL_PRECISION_MODEL_DIR
TEMPLATE_NVFP4_DIR=$TEMPLATE_NVFP4_DIR
OUTPUT_DIR=$OUTPUT_DIR
PRESET=$PRESET
LAYERWISE_PROFILE=$LAYERWISE_PROFILE
LAYERWISE_RULES_JSON=$LAYERWISE_RULES_JSON
DEVICE=$DEVICE
MAX_SHARD_SIZE=$MAX_SHARD_SIZE
LOG_FILE=$LOG_FILE
EOF

export PYTHONPATH="$REPO_ROOT"

args=(
  --full-precision-model-dir "$FULL_PRECISION_MODEL_DIR"
  --template-nvfp4-dir "$TEMPLATE_NVFP4_DIR"
  --output-dir "$OUTPUT_DIR"
  --preset "$PRESET"
  --layerwise-profile "$LAYERWISE_PROFILE"
  --device "$DEVICE"
  --max-shard-size "$MAX_SHARD_SIZE"
)

if [[ -n "$LAYERWISE_RULES_JSON" ]]; then
  args+=(--layerwise-rules-json "$LAYERWISE_RULES_JSON")
fi

"$PYTHON_BIN" -u \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/export_global_budget_repo_mse_sweep.py" \
  "${args[@]}" \
  2>&1 | tee "$LOG_FILE"

cat <<EOF
EXPORT_DONE=1
OUTPUT_DIR=$OUTPUT_DIR
REPORT_JSON=$OUTPUT_DIR/global_budget_repo_mse_sweep_export.json
LOG_FILE=$LOG_FILE
EOF
