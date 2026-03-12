#!/usr/bin/env bash
set -euo pipefail

HOME="${HOME:-/sim/eec/shared/junfu.qx}"
MODEL_DIR="${MODEL_DIR:-/sim/eec/shared/models/Qwen}"
REPO_ROOT="${REPO_ROOT:-$HOME/Model-Optimizer}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Qwen3-4B-NVFP4-compress10p26}"
FP_MODEL_DIR="${FP_MODEL_DIR:-$MODEL_DIR/Qwen3-4B}"
TEMPLATE_DIR="${TEMPLATE_DIR:-$MODEL_DIR/Qwen3-4B-NVFP4}"
HF_REPO_ID="${HF_REPO_ID:-DrQianXu/Qwen3-4B-nvfp4-Compressible}"

"$PYTHON_BIN" - <<PY
import shutil
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
if output_dir.exists():
    shutil.rmtree(output_dir)
    print(f"removed stale output: {output_dir}")
PY

export PYTHONPATH="$REPO_ROOT"

"$PYTHON_BIN" \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/export_double_scale_repo_mse_sweep.py" \
  --full-precision-model-dir "$FP_MODEL_DIR" \
  --template-nvfp4-dir "$TEMPLATE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --alpha 2.0 \
  --min-scale-ratio 1.0 \
  --soft-entropy-budget-ratio 0.26 \
  --soft-temperature 0.20 \
  --candidate-chunk-size 32 \
  --block-chunk-size 65536 \
  --max-shard-size 1GB

cat > "$OUTPUT_DIR/README.md" <<'EOF'
---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-4B
tags:
- nvfp4
- quantization
- modelopt
- qwen3
---

# Qwen3-4B NVFP4 Compressible

This model is a standard NVFP4 checkpoint derived from `Qwen/Qwen3-4B`, exported with a constrained `double-scale + repo-style FP8 MSE sweep` procedure.

Method summary:
- Start from the BF16 source weights from `Qwen/Qwen3-4B`
- Use the existing `Qwen3-4B-NVFP4` checkpoint as the HF/template layout
- Inflate the layer-global NVFP4 scale by `alpha=2.0`
- Run repo-style per-block FP8 E4M3 MSE sweep under the inflated global bound
- Add a soft-code entropy budget constraint with `soft_entropy_budget_ratio=0.26`
- Export back to standard `weight_packed / weight_scale / weight_global_scale`

Representative-layer tradeoff before full export:
- Baseline absmax NVFP4 compression proxy: `0.89%`
- Direct double-scale compression proxy: `12.07%`
- Final constrained sweep compression proxy: `10.07%`
- Baseline absmax weight MSE: `4.62e-6`
- Final constrained sweep weight MSE: `5.94e-6`

Notes:
- This is still a standard NVFP4 checkpoint for downstream tooling.
- The compression metric above is the byte-entropy proxy on `weight_packed`, not BF16-to-NVFP4 size ratio.
- The full export script lives in `experimental/nvfp4_scale_inflation/export_double_scale_repo_mse_sweep.py`.
EOF

"$PYTHON_BIN" \
  "$REPO_ROOT/experimental/nvfp4_scale_inflation/upload_hf_folder.py" \
  --folder-path "$OUTPUT_DIR" \
  --repo-id "$HF_REPO_ID" \
  --commit-message "Replace with constrained double-scale NVFP4 export (~10% compression proxy)"
