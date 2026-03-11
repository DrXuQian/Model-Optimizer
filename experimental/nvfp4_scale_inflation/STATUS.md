# NVFP4 Scale Inflation Status

## Current Method

The current method keeps the checkpoint in standard compressed-tensors NVFP4 format:

- `weight_packed`
- `weight_scale`
- `weight_global_scale`

The transform is:

1. Start from an existing NVFP4 checkpoint.
2. Read `weight_packed`, `weight_scale`, and `weight_global_scale`.
3. Decode per-block effective scales.
4. Inflate scales by a constant `alpha`.
5. Re-quantize back to standard NVFP4 packed bytes.
6. Optionally refine scales with Adam + STE under the constraint `scale >= original_scale`.

The implementation is in:

- [scale_inflation.py](scale_inflation.py)
- [export_checkpoint.py](export_checkpoint.py)
- [export_from_full_precision.py](export_from_full_precision.py)
- [eval_mmlu_batched.py](eval_mmlu_batched.py)

## Overall Summary

| Variant | Weight Source | Compression Proxy | Reference MSE | MMLU Sample | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline `Qwen3-4B-NVFP4` | Existing NVFP4 | `0.96%` | `4.96e-6` vs BF16 | `29 / 40 = 72.5%` | Structural baseline |
| `alpha=2.0` on existing NVFP4 | Existing NVFP4 | `12.70%` | `4.92e-6` vs dequantized NVFP4 | `27 / 40 = 67.5%` | Better compressibility, worse sample accuracy |
| `alpha=2.0` from BF16 source | `Qwen3-4B` | `12.07%` | `7.46e-6` vs BF16 | `30 / 40 = 75.0%` | Pure NVFP4, best current accuracy |

## Existing Results

### Checkpoint Used

- Baseline NVFP4 checkpoint: `Qwen3-4B-NVFP4` (local experiment input, not tracked in git)

### Compression Result

| Experiment | Layers | Baseline Compression Proxy | Alpha Compression Proxy | Optimized Compression Proxy | Reference MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-model alpha-only, `alpha=2.0` | 252 | `0.96%` | `12.70%` | `-` | `4.92e-6` vs dequantized NVFP4 |
| Representative run, `alpha=2.0` + 200-step Adam | 4 | `0.92%` | `12.58%` | `15.86%` | `4.57e-6` vs dequantized NVFP4 |

Artifacts:

- [../../outputs/nvfp4_scale_inflation_qwen3_all_alpha.json](../../outputs/nvfp4_scale_inflation_qwen3_all_alpha.json)
- [../../outputs/nvfp4_scale_inflation_qwen3_repr_opt.json](../../outputs/nvfp4_scale_inflation_qwen3_repr_opt.json)

### Accuracy Result

Because full MMLU on one RTX 5070 through HF + compressed-tensors was too slow, the current accuracy check is a fixed MMLU sample:

| Setting | Value |
| --- | --- |
| Subjects | `8` representative subjects |
| Samples per subject | `5` |
| Prompt format | `5-shot` |
| Total questions | `40` |
| Subject list | `abstract_algebra`, `college_computer_science`, `clinical_knowledge`, `miscellaneous`, `econometrics`, `sociology`, `philosophy`, `high_school_world_history` |

Result on the 40-question sample:

| Model | Correct / Total | Accuracy | Delta vs Baseline |
| --- | ---: | ---: | ---: |
| Baseline NVFP4 | `29 / 40` | `72.5%` | `-` |
| `alpha=2.0` on existing NVFP4 | `27 / 40` | `67.5%` | `-2` questions |

Artifacts:

- [../../outputs/mmlu_batched_sample8_base.json](../../outputs/mmlu_batched_sample8_base.json)
- [../../outputs/mmlu_batched_sample8_alpha2.json](../../outputs/mmlu_batched_sample8_alpha2.json)

## Full-Precision Source Experiment

To avoid optimizing against already-quantized weights, the next step uses the original source checkpoint:

- `Qwen3-4B` (local BF16 experiment input, not tracked in git)

This path still exports a pure NVFP4 checkpoint. No mixed-precision format was used.

### Export Strategy

1. Use the existing NVFP4 checkpoint as the structural/template baseline:
   `Qwen3-4B-NVFP4`
2. Read the original BF16 `*.weight` tensors from
   `Qwen3-4B`
3. Re-quantize those weights into standard NVFP4 using inflated scales derived from the baseline NVFP4 scales.
4. Keep the exported format identical to compressed-tensors NVFP4.

Exported checkpoint:

- `Qwen3-4B-NVFP4-frombf16-alpha2` (local exported checkpoint, not tracked in git)

### Compression / MSE vs Full-Precision Weights

| Variant | Weighted Compression Proxy | Weighted MSE to BF16 Weights |
| --- | ---: | ---: |
| Baseline NVFP4 template | `0.96%` | `4.96e-6` |
| `alpha=2.0` from BF16 source | `12.07%` | `7.46e-6` |

Artifact:

- [../../outputs/nvfp4_scale_inflation_from_full_precision_export.json](../../outputs/nvfp4_scale_inflation_from_full_precision_export.json)

### Accuracy vs Baseline

| Model | Correct / Total | Accuracy | Delta vs Baseline |
| --- | ---: | ---: | ---: |
| Baseline NVFP4 | `29 / 40` | `72.5%` | `-` |
| `alpha=2.0` on existing NVFP4 | `27 / 40` | `67.5%` | `-2` questions |
| `alpha=2.0` from BF16 source | `30 / 40` | `75.0%` | `+1` question |

Per-subject breakdown on the same 8-subject sample:

| Subject | Baseline NVFP4 | `alpha=2.0` on Existing NVFP4 | `alpha=2.0` from BF16 |
| --- | ---: | ---: | ---: |
| `miscellaneous` | `0.6` | `0.6` | `0.8` |
| `philosophy` | `0.8` | `0.8` | `0.8` |
| `abstract_algebra` | `0.2` | `0.2` | `0.2` |
| `clinical_knowledge` | `0.8` | `0.8` | `0.8` |
| `sociology` | `0.8` | `0.8` | `1.0` |
| `econometrics` | `1.0` | `0.8` | `1.0` |
| `college_computer_science` | `0.6` | `0.6` | `0.6` |
| `high_school_world_history` | `1.0` | `0.8` | `0.8` |

Artifact:

- [../../outputs/mmlu_batched_sample8_frombf16_alpha2.json](../../outputs/mmlu_batched_sample8_frombf16_alpha2.json)

## Test Cases

### Unit Tests

- [../../tests/unit/torch/quantization/test_nvfp4_scale_inflation.py](../../tests/unit/torch/quantization/test_nvfp4_scale_inflation.py)

Covered checks:

- FP4 nibble pack/unpack roundtrip
- checkpoint dequantize/requantize roundtrip
- scale encoding shape and positivity
- STE lower-bound constraint
- compression metric shape/range

Run command:

```bash
conda run --no-capture-output -n modelopt python -m pytest -q -o addopts='' \
  tests/unit/torch/quantization/test_nvfp4_scale_inflation.py
```

Result:
| Test File | Result |
| --- | --- |
| [../../tests/unit/torch/quantization/test_nvfp4_scale_inflation.py](../../tests/unit/torch/quantization/test_nvfp4_scale_inflation.py) | `5 passed` |

### Accuracy Script

The sample MMLU evaluation used:

```bash
conda run --no-capture-output -n modelopt python -m experimental.nvfp4_scale_inflation.eval_mmlu_batched \
  --model-path <checkpoint_dir> \
  --output-json <report.json> \
  --batch-size 8 \
  --limit-per-subject 5 \
  --subjects abstract_algebra,college_computer_science,clinical_knowledge,miscellaneous,econometrics,sociology,philosophy,high_school_world_history
```

## Limitation

The current checkpoint-based method optimizes against dequantized NVFP4 weights, not against the original BF16 checkpoint. This is why the next step is to use the full-precision source model:

- `Qwen3-4B`
