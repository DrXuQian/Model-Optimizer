#!/usr/bin/env bash
set -euo pipefail

HOME="${HOME:-/sim/eec/shared/junfu.qx}"
MODEL_DIR="${MODEL_DIR:-/sim/eec/shared/models/Qwen}"
REPO_ROOT="${REPO_ROOT:-$HOME/Model-Optimizer}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Qwen3-4B-NVFP4-global-budget}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload global-budget NVFP4 export}"

"$REPO_ROOT/experimental/nvfp4_scale_inflation/run_export_global_budget_hf.sh"

if [[ -z "$HF_REPO_ID" ]]; then
  echo "HF_REPO_ID is empty; export finished locally at $OUTPUT_DIR"
  exit 0
fi

args=(
  --folder-path "$OUTPUT_DIR"
  --repo-id "$HF_REPO_ID"
  --commit-message "$HF_COMMIT_MESSAGE"
)

if [[ "$HF_PRIVATE" == "1" ]]; then
  args+=(--private)
fi

export PYTHONPATH="$REPO_ROOT"
"$PYTHON_BIN" \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/upload_hf_folder.py" \
  "${args[@]}"

echo "HF_UPLOAD_DONE=1"
echo "HF_REPO_ID=$HF_REPO_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
