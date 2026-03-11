#!/usr/bin/env bash
set -euo pipefail

# Experimental H800 sweep for weight-only NVFP4 QAT with soft-code compression regularization.
#
# This script assumes a baseline checkpoint already exists. It launches several resumed
# compression-aware runs with different entropy targets and penalty strengths.
#
# Example:
#   BASELINE_DIR=/path/to/baseline_qat \
#   OUTPUT_ROOT=/path/to/out \
#   bash examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800_sweep.sh

BASELINE_DIR="${BASELINE_DIR:-/home/qianxu/TensorRT-Model-Optimizer/outputs/qwen3_4b_nvfp4_soft_code_qat/baseline_qat}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/qianxu/TensorRT-Model-Optimizer/outputs/qwen3_4b_nvfp4_soft_code_qat/sweeps}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_SIZE="${TRAIN_SIZE:-4096}"
EVAL_SIZE="${EVAL_SIZE:-512}"
CALIB_SIZE="${CALIB_SIZE:-512}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-500}"
LR="${LR:-1e-5}"

TARGET_RATIOS="${TARGET_RATIOS:-0.98 0.95 0.92 0.90}"
PENALTIES="${PENALTIES:-0.1 1.0}"
DUAL_LRS="${DUAL_LRS:-0.01 0.05}"
SAMPLE_LAYERS="${SAMPLE_LAYERS:-4 8}"
SAMPLE_BLOCKS="${SAMPLE_BLOCKS:-128 256}"

mkdir -p "${OUTPUT_ROOT}"

run_id=0
for ratio in ${TARGET_RATIOS}; do
  for rho in ${PENALTIES}; do
    for dual_lr in ${DUAL_LRS}; do
      for sample_layers in ${SAMPLE_LAYERS}; do
        for sample_blocks in ${SAMPLE_BLOCKS}; do
          run_id=$((run_id + 1))
          out_dir="${OUTPUT_ROOT}/run_${run_id}_r${ratio}_rho${rho}_dlr${dual_lr}_sl${sample_layers}_sb${sample_blocks}"
          echo "Launching ${out_dir}"
          "${PYTHON_BIN}" examples/llm_qat/main.py \
            --model_name_or_path "${BASELINE_DIR}" \
            --quant_cfg NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG \
            --calib_size "${CALIB_SIZE}" \
            --train_size "${TRAIN_SIZE}" \
            --eval_size "${EVAL_SIZE}" \
            --model_max_length "${MODEL_MAX_LENGTH}" \
            --per_device_train_batch_size "${BATCH_SIZE}" \
            --per_device_eval_batch_size "${BATCH_SIZE}" \
            --gradient_accumulation_steps "${GRAD_ACCUM}" \
            --max_steps "${MAX_STEPS}" \
            --learning_rate "${LR}" \
            --do_train True \
            --do_eval False \
            --gradient_checkpointing True \
            --logging_steps 10 \
            --save_strategy steps \
            --save_steps 100 \
            --eval_strategy no \
            --report_to none \
            --compression_loss True \
            --compression_proxy_type soft_code \
            --compression_target_ratio "${ratio}" \
            --compression_penalty_rho "${rho}" \
            --compression_dual_lr "${dual_lr}" \
            --compression_sample_layers "${sample_layers}" \
            --compression_sample_blocks_per_layer "${sample_blocks}" \
            --compression_temperature 0.25 \
            --output_dir "${out_dir}"
        done
      done
    done
  done
done

echo "Completed ${run_id} compression-aware sweep runs."
