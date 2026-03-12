#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/qianxu/TensorRT-Model-Optimizer}"
ENV_NAME="${ENV_NAME:-modelopt}"
FULL_PRECISION_MODEL_DIR="${FULL_PRECISION_MODEL_DIR:-$REPO_ROOT/Qwen3-4B}"
LAYERS="${LAYERS:-model.layers.0.self_attn.q_proj,model.layers.0.self_attn.k_proj,model.layers.0.self_attn.v_proj,model.layers.0.self_attn.o_proj}"
ROW_LIMIT="${ROW_LIMIT:-128}"
PRESET="${PRESET:-compress10}"
JOBS="${JOBS:-2}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_ROOT/outputs/global_budget_attention_sweep_${PRESET}_row${ROW_LIMIT}.json}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/global_budget_attention_sweep_${PRESET}_row${ROW_LIMIT}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

cat <<EOF
REPO_ROOT=$REPO_ROOT
FULL_PRECISION_MODEL_DIR=$FULL_PRECISION_MODEL_DIR
LAYERS=$LAYERS
ROW_LIMIT=$ROW_LIMIT
PRESET=$PRESET
JOBS=$JOBS
OUTPUT_JSON=$OUTPUT_JSON
LOG_FILE=$LOG_FILE
EOF

export PYTHONPATH="$REPO_ROOT"

conda run --no-capture-output -n "$ENV_NAME" python -u \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/global_budget_layer_sweep.py" \
  --full-precision-model-dir "$FULL_PRECISION_MODEL_DIR" \
  --layers "$LAYERS" \
  --row-limit "$ROW_LIMIT" \
  --preset "$PRESET" \
  --jobs "$JOBS" \
  --output-json "$OUTPUT_JSON" \
  2>&1 | tee "$LOG_FILE"
