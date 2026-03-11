# NVFP4 Scale Inflation

Experimental utilities for testing NVFP4 scale inflation on exported checkpoints.

## What It Does

This experiment:

- loads an NVFP4 checkpoint saved in safetensors format
- reconstructs dequantized weights from `weight_packed`, `weight_scale`, and `weight_global_scale`
- inflates per-block scales by a constant `alpha`
- re-quantizes weights back to standard NVFP4
- optionally refines per-block scales with Adam while enforcing `scale >= original_scale`
- reports entropy-based compressibility and MSE against the reconstructed baseline

There is also a full-precision source path:

- start from a full-precision HF checkpoint
- use an existing NVFP4 checkpoint only as a structural/template baseline
- re-quantize the original weights back into standard compressed-tensors NVFP4

## Important Limitation

If the checkpoint does not contain the original FP16/BF16/FP32 weights, the experiment uses the
dequantized NVFP4 weights as the optimization target. This measures whether scale inflation can make
the existing NVFP4 checkpoint more compressible, not whether it improves over the original
full-precision model.

## Usage

```bash
conda run -n modelopt python -m experimental.nvfp4_scale_inflation.scale_inflation \
  --checkpoint Qwen3-4B-NVFP4/model.safetensors \
  --alpha 2.0 \
  --steps 200 \
  --lr 1e-3 \
  --device cuda \
  --max-layers 252 \
  --optimize-max-layers 4 \
  --output-json outputs/nvfp4_scale_inflation_qwen3_4b.json
```

## Output

The script prints a per-layer summary and writes a JSON report containing:

- baseline entropy/compression metrics
- alpha-only metrics
- optimized metrics when enabled
- scale ratio statistics
- aggregate weighted averages across all processed layers

## Full-Precision Export

```bash
conda run -n modelopt python -m experimental.nvfp4_scale_inflation.export_from_full_precision \
  --full-precision-model-dir Qwen3-4B \
  --template-nvfp4-dir Qwen3-4B-NVFP4 \
  --output-dir Qwen3-4B-NVFP4-frombf16-alpha2 \
  --alpha 2.0 \
  --optimize-max-layers 0 \
  --device cpu
```

## Reference Results

- Summary report: [STATUS.md](STATUS.md)
- Existing-NVFP4 compression report: [../../outputs/nvfp4_scale_inflation_qwen3_all_alpha.json](../../outputs/nvfp4_scale_inflation_qwen3_all_alpha.json)
- Existing-NVFP4 optimization report: [../../outputs/nvfp4_scale_inflation_qwen3_repr_opt.json](../../outputs/nvfp4_scale_inflation_qwen3_repr_opt.json)
- MMLU sample baseline: [../../outputs/mmlu_batched_sample8_base.json](../../outputs/mmlu_batched_sample8_base.json)
- MMLU sample alpha on existing NVFP4: [../../outputs/mmlu_batched_sample8_alpha2.json](../../outputs/mmlu_batched_sample8_alpha2.json)
- MMLU sample alpha from BF16 source: [../../outputs/mmlu_batched_sample8_frombf16_alpha2.json](../../outputs/mmlu_batched_sample8_frombf16_alpha2.json)
- BF16-source export summary: [../../outputs/nvfp4_scale_inflation_from_full_precision_export.json](../../outputs/nvfp4_scale_inflation_from_full_precision_export.json)
