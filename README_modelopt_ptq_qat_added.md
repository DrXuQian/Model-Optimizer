# 新增代码说明：PTQ / QAT 支持与命令行

本文档聚焦本次新增的两块能力：

1. 在 `evalscope` 中新增 `modelopt_hf` 后端，支持基于 HuggingFace checkpoint 的进程内 ModelOpt 量化评测。
2. 在 `ms-swift` 中新增 `--enable_modelopt_hf` 开关，支持加载包含 `modelopt_state.pth` 的 ModelOpt checkpoint 并继续训练。

---

## 1. 新增代码位置与作用

### 1.1 EvalScope：`modelopt_hf` 评测后端

- `evalscope/evalscope/models/modelopt_hf.py`
  - 新增 `ModelOptHFAPI`。
  - 作用：在加载 HF 模型后，按 `quant_cfg` 执行 ModelOpt 量化（如 `NVFP4_DEFAULT_CFG`），然后走 EvalScope 任务评测流程。
  - 支持通过 `quant_cfg_overrides` 细粒度关闭某些模块量化（例如视觉分支）。

- `evalscope/evalscope/models/model_apis.py`
  - 注册 `modelopt_hf` API。

- `evalscope/evalscope/constants.py`
  - 新增 `EvalType.MODELOPT_HF`。

- `evalscope/examples/example_eval_modelopt_nvfp4_hf.py`
  - 提供直接可运行的示例命令入口。

### 1.2 ms-swift：加载 ModelOpt checkpoint 继续训练

- `ms-swift/swift/arguments/base_args/model_args.py`
  - 新增参数：`enable_modelopt_hf: bool = False`。

- `ms-swift/swift/arguments/base_args/base_args.py`
  - 支持从 checkpoint 参数恢复 `enable_modelopt_hf`。

- `ms-swift/swift/model/register.py`
  - 当 `enable_modelopt_hf=True` 时，在 `from_pretrained` 之前调用：
    - `mto.enable_huggingface_checkpointing()`
  - 作用：自动恢复 `modelopt_state.pth`，避免只加载权重导致量化结构状态缺失。

- `ms-swift/scripts/smoke_modelopt_hf.sh`
  - 新增一键 smoke 脚本，用于回归验证：
    - 量化 -> 保存 `modelopt_state.pth` -> 用 ms-swift 加载恢复 -> 单步训练。

---

## 2. PTQ 命令行

### 2.1 量化导出（ModelOpt PTQ）

在本仓库执行：

```bash
conda activate modelopt
python examples/llm_ptq/hf_ptq.py \
  --pyt_ckpt_path /path/to/your_hf_model \
  --qformat nvfp4 \
  --calib_size 16 \
  --calib_seq 512 \
  --export_path outputs/ptq_nvfp4_model \
  --trust_remote_code
```

常用说明：

- `--qformat nvfp4`：选择 NVFP4 PTQ。
- `--calib_size`：校准样本数量。
- `--export_path`：导出路径（HF 格式）。

### 2.2 量化后评测（EvalScope + 新增 `modelopt_hf`）

在 `evalscope` 仓库执行：

```bash
cd evalscope
conda activate modelopt
python examples/example_eval_modelopt_nvfp4_hf.py \
  --model_path /path/to/your_hf_model_or_ptq_model \
  --dataset gsm8k \
  --quant_cfg NVFP4_DEFAULT_CFG \
  --calib_size 16 \
  --limit 50
```

如果是 VLM/Omni 并希望跳过视觉分支量化：

```bash
python examples/example_eval_modelopt_nvfp4_hf.py \
  --model_path /path/to/your_vlm_model \
  --dataset gsm8k \
  --quant_cfg NVFP4_DEFAULT_CFG \
  --calib_size 16 \
  --disable_vision_quant \
  --limit 50
```

---

## 3. QAT 命令行

### 3.1 本仓库 QAT（推荐基线流程）

在本仓库执行（`examples/llm_qat/launch.sh`）：

```bash
cd examples/llm_qat
conda activate modelopt

# Step 1: 先训练 BF16 基线
./launch.sh \
  --model meta-llama/Meta-Llama-3-8B \
  --do_train True \
  --num_epochs 1 \
  --lr 1e-5 \
  --output_dir outputs/llama3_finetune

# Step 2: 在基线上做 QAT（NVFP4）
./launch.sh \
  --model outputs/llama3_finetune \
  --do_train True \
  --num_epochs 1 \
  --lr 1e-5 \
  --quant_cfg NVFP4_DEFAULT_CFG \
  --calib_size 512 \
  --output_dir outputs/llama3_qat_nvfp4
```

说明：

- `--quant_cfg NVFP4_DEFAULT_CFG`：开启 NVFP4 QAT。
- `--do_train True`：执行训练（QAT阶段）。

### 3.2 ms-swift 中继续训练 ModelOpt checkpoint（新增能力）

如果你已有包含 `modelopt_state.pth` 的 checkpoint，希望在 ms-swift 继续训练，需要开启：

- `--enable_modelopt_hf true`

最小回归方式（新增脚本）：

```bash
cd ms-swift
scripts/smoke_modelopt_hf.sh --conda-env modelopt
```

该脚本会自动完成：

1. 量化模型并保存 `modelopt_state.pth`
2. 使用 ms-swift 并开启 `enable_modelopt_hf` 重新加载
3. 运行 1 个训练 step 验证可训练性

---

## 4. 你该怎么选

- 只做部署前量化：优先 PTQ（`examples/llm_ptq/hf_ptq.py`）。
- 低比特精度掉点明显：走 QAT（`examples/llm_qat/launch.sh`）。
- 需要在 ms-swift 接着训 ModelOpt checkpoint：务必打开 `--enable_modelopt_hf true`。

---

## 5. 常见问题

### Q1：为什么要恢复 `modelopt_state.pth`？

因为权重文件本身不完整描述 ModelOpt 的量化状态与结构修改。只加载权重而不恢复 `modelopt_state`，会导致“看起来加载成功，但量化状态不完整”。

### Q2：本地 checkpoint 加载时 `model_type` 二义性怎么办？

在 ms-swift 显式传 `model_type`（例如 `qwen2`），避免自动匹配歧义。
