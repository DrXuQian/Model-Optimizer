# Experimental Compression-Aware NVFP4 QAT

This directory contains an experimental QAT path that keeps the standard ModelOpt NVFP4 fake-quant flow and adds an extra compression proxy loss on top of the trainable weights.

Current v1 design:

- Weight quantization init: `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG`
- Trainable variables: model weights only
- Compression proxy: `soft_code`
- Future hook: `soft_byte` proxy type is reserved but not implemented yet
- Static NVFP4 weight quantizers are required. `NVFP4_DEFAULT_CFG` is not supported because its
  weight quantizer is dynamic.

Files:

- `compression_proxy.py`: compression proxy utilities
- `compression_aware_trainer.py`: trainer subclasses that add the proxy loss
- `run_qwen3_4b_nvfp4_soft_code_h800.sh`: single-GPU H800 recipe

Key flags in `main.py`:

- `--compression_loss True`
- `--compression_proxy_type soft_code`
- `--compression_target_ratio 0.90`
- `--compression_penalty_rho 1.0`
- `--compression_dual_lr 0.05`
- `--compression_sample_layers 8`
- `--compression_sample_blocks_per_layer 256`
- `--compression_temperature 0.25`

Recommended first experiment:

1. Run baseline QAT with `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG`
2. Resume from that checkpoint with `--compression_loss True`
3. Compare:
   - task accuracy
   - hard NVFP4 entropy/compression proxy on exported weights
   - real compressed checkpoint size

Night plan / H800 plan:

1. Run baseline QAT to a stable checkpoint.
2. Resume from baseline with `soft_code` enabled and sweep:
   - `compression_target_ratio`: `0.98`, `0.95`, `0.92`, `0.90`
   - `compression_penalty_rho`: `0.1`, `1.0`
   - `compression_dual_lr`: `0.01`, `0.05`
   - `compression_sample_layers`: `4`, `8`
   - `compression_sample_blocks_per_layer`: `128`, `256`
3. For each checkpoint, export and measure:
   - hard code histogram entropy
   - user-defined 32KB even/odd byte entropy proxy
   - actual compressed checkpoint size with `zstd`
4. Keep only the best 1 to 2 candidates for MMLU.
