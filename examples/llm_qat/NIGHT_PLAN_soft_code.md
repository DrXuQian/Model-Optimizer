# Night Plan: Soft-Code Compression-Aware NVFP4 QAT

## Goal

Implement and validate a first compression-aware QAT path that:

- keeps standard ModelOpt NVFP4 fake quantization,
- tunes model weights only,
- regularizes a differentiable soft-code histogram entropy proxy,
- leaves a clean interface for future `soft_byte` work.

## Current v1 Scope

- Supported quantization init:
  - `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG`
  - `NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`
- Unsupported for v1:
  - `NVFP4_DEFAULT_CFG`
  - any dynamic NVFP4 weight config
- Trainable parameters:
  - base model weights only
- Compression loss:
  - `soft_code` entropy proxy
- Future upgrade path:
  - `soft_byte`

## What Is Implemented

- `examples/llm_qat/compression_proxy.py`
  - `soft_code` entropy proxy
  - weight tensor sampling
  - interface placeholder for `soft_byte`
- `examples/llm_qat/compression_aware_trainer.py`
  - trainer mixin
  - direct entropy regularization
  - augmented-Lagrangian-style target tracking
- `examples/llm_qat/main.py`
  - new CLI arguments
  - trainer routing
  - explicit static-NVFP4 config check
- `examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800.sh`
  - baseline + resumed compression-aware run
- `examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800_sweep.sh`
  - H800 sweep recipe

## Immediate Validation Tasks

1. Complete one end-to-end smoke test with:
   - `Qwen3-4B`
   - `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG`
   - `compression_loss=True`
   - `max_steps=1`
2. Verify training logs contain:
   - `compression_entropy_bits`
   - `compression_regularizer`
   - `compression_num_layers`
   - `compression_num_blocks`
3. Save a tiny checkpoint and confirm it restores correctly.

## H800 Experiments for Tomorrow

### Phase 1: Baseline

Run:

- `examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800.sh`

Keep:

- baseline checkpoint
- training logs
- final eval logs

### Phase 2: Compression-Aware Resume

Primary sweep:

- `compression_target_ratio`: `0.98`, `0.95`, `0.92`, `0.90`
- `compression_penalty_rho`: `0.1`, `1.0`
- `compression_dual_lr`: `0.01`, `0.05`
- `compression_sample_layers`: `4`, `8`
- `compression_sample_blocks_per_layer`: `128`, `256`

Run:

- `examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800_sweep.sh`

### Phase 3: Export + Hard Metrics

For each candidate:

1. export checkpoint
2. compute hard FP4 code histogram
3. compute hard 32KB even/odd byte entropy using the existing experimental NVFP4 tools
4. compress with `zstd`
5. record actual file size

### Phase 4: Accuracy

For top 1 to 2 candidates:

1. run large-sample MMLU
2. compare against baseline QAT checkpoint
3. compare against the existing published NVFP4 baseline

## Success Criteria

The v1 line is worth continuing only if at least one candidate satisfies:

- no obvious training instability,
- hard compression proxy improves over baseline,
- task accuracy degradation is acceptable,
- actual compressed checkpoint size also improves.

## Next Upgrade After v1

If v1 is promising:

1. implement `soft_byte`
2. compare `soft_code` vs `soft_byte`
3. consider adding scale training only if weight-only QAT saturates
