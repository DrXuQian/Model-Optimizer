#!/bin/bash
# Export NVFP4 model with hard-code entropy proxy
# Target: ~90% compression rate (controlled by CIP-Proxy selection, not entropy budget)
#
# Usage:
#   bash experimental/nvfp4_scale_inflation/run_export_hardcode_vs_softcode.sh
#
# After export, push to HuggingFace:
#   python experimental/nvfp4_scale_inflation/upload_hf_folder.py \
#     --folder /root/autodl-tmp/Qwen3-4B-NVFP4-hardcode \
#     --repo-id <your-hf-repo>

set -e

cd /root/Model-Optimizer

# Clean previous output if exists
rm -rf /root/autodl-tmp/Qwen3-4B-NVFP4-hardcode

echo "=== Exporting with hard-code entropy proxy ==="
echo "Preset: compress10 (alpha=2.0, soft_entropy_budget_ratio=0.25)"
echo "Entropy proxy: hard"
echo ""

python3 experimental/nvfp4_scale_inflation/export_global_budget_repo_mse_sweep.py \
  --full-precision-model-dir /root/autodl-tmp/Qwen3-4B \
  --template-nvfp4-dir /root/autodl-tmp/Qwen3-4B-NVFP4 \
  --output-dir /root/autodl-tmp/Qwen3-4B-NVFP4-hardcode \
  --preset compress10 \
  --device cuda \
  --entropy-proxy hard

echo ""
echo "=== Done ==="
echo "Hard-code model: /root/autodl-tmp/Qwen3-4B-NVFP4-hardcode"
echo ""
echo "Check report:"
echo "  cat /root/autodl-tmp/Qwen3-4B-NVFP4-hardcode/global_budget_repo_mse_sweep_export.json | python3 -m json.tool | grep weighted"
