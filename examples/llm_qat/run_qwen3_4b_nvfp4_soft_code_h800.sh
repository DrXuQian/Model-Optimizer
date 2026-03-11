#!/usr/bin/env bash
set -euo pipefail

# Experimental single-GPU H800 recipe for Qwen3-4B NVFP4 QAT with soft-code compression loss.
#
# Usage:
#   bash examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800.sh
#
# Override any variable inline, e.g.:
#   MODEL_PATH=/path/to/Qwen3-4B OUTPUT_ROOT=/tmp/qwen3_nvfp4 bash examples/llm_qat/run_qwen3_4b_nvfp4_soft_code_h800.sh

MODEL_PATH="${MODEL_PATH:-/home/qianxu/TensorRT-Model-Optimizer/Qwen3-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/qianxu/TensorRT-Model-Optimizer/outputs/qwen3_4b_nvfp4_soft_code_qat}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_SIZE="${TRAIN_SIZE:-4096}"
EVAL_SIZE="${EVAL_SIZE:-512}"
CALIB_SIZE="${CALIB_SIZE:-512}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS_BASELINE="${MAX_STEPS_BASELINE:-500}"
MAX_STEPS_COMPRESSION="${MAX_STEPS_COMPRESSION:-500}"
LR="${LR:-1e-5}"

BASELINE_DIR="${OUTPUT_ROOT}/baseline_qat"
COMPRESSION_DIR="${OUTPUT_ROOT}/compression_qat"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage 1: baseline NVFP4 QAT"
"${PYTHON_BIN}" examples/llm_qat/main.py \
  --model_name_or_path "${MODEL_PATH}" \
  --quant_cfg NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG \
  --calib_size "${CALIB_SIZE}" \
  --train_size "${TRAIN_SIZE}" \
  --eval_size "${EVAL_SIZE}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_steps "${MAX_STEPS_BASELINE}" \
  --learning_rate "${LR}" \
  --do_train True \
  --do_eval False \
  --gradient_checkpointing True \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 100 \
  --eval_strategy no \
  --report_to none \
  --output_dir "${BASELINE_DIR}"

echo "Stage 2: compression-aware NVFP4 QAT"
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
  --max_steps "${MAX_STEPS_COMPRESSION}" \
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
  --compression_target_ratio 0.90 \
  --compression_penalty_rho 1.0 \
  --compression_dual_lr 0.05 \
  --compression_sample_layers 8 \
  --compression_sample_blocks_per_layer 256 \
  --compression_temperature 0.25 \
  --output_dir "${COMPRESSION_DIR}"

echo "Done."
echo "Baseline checkpoint: ${BASELINE_DIR}"
echo "Compression-aware checkpoint: ${COMPRESSION_DIR}"
