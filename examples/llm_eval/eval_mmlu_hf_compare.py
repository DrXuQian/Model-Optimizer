# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare MMLU accuracy between a full-precision HF checkpoint and an NVFP4 HF checkpoint.

Example:
    conda run --no-capture-output -n modelopt python examples/llm_eval/eval_mmlu_hf_compare.py \
        --fp-model-path /home/qianxu/TensorRT-Model-Optimizer/Qwen3-4B \
        --nvfp4-model-path /home/qianxu/TensorRT-Model-Optimizer/Qwen3-4B-NVFP4 \
        --output-dir /home/qianxu/TensorRT-Model-Optimizer/outputs/mmlu_qwen3_compare \
        --batch-size 8 \
        --ntrain 5
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import modelopt.torch.opt as mto

    mto.enable_huggingface_checkpointing()
except Exception:
    mto = None


CHOICE_LABELS = ["A", "B", "C", "D"]
CHOICE_TOKENS = [" A", " B", " C", " D"]
DEFAULT_MAX_SEQ_LEN = 4096


@dataclass
class EvalExample:
    example_id: str
    subject: str
    prompt: str
    answer: int
    prompt_length: int


@dataclass
class EvalResult:
    label: str
    model_path: str
    dtype: str
    accuracy: float
    num_examples: int
    runtime_sec: float
    per_subject: dict[str, dict[str, float | int]]
    predictions: dict[str, int]
    correctness: dict[str, bool]


def format_subject(subject: str) -> str:
    return " ".join(subject.split("_"))


def format_example(question: str, choices: list[str], answer: int | None = None) -> str:
    prompt = question
    for index, choice in enumerate(choices):
        prompt += f"\n{CHOICE_LABELS[index]}. {choice}"
    prompt += "\nAnswer:"
    if answer is not None:
        prompt += f" {CHOICE_LABELS[answer]}\n\n"
    return prompt


def build_fewshot_prefixes(dev_rows: list[dict], subject: str) -> list[str]:
    header = (
        f"The following are multiple choice questions (with answers) about "
        f"{format_subject(subject)}.\n\n"
    )
    prefixes = [header]
    text = header
    for row in dev_rows:
        text += format_example(row["question"], row["choices"], int(row["answer"]))
        prefixes.append(text)
    return prefixes


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text).input_ids)


def build_eval_examples(
    *,
    tokenizer,
    test_rows_by_subject: dict[str, list[dict]],
    dev_rows_by_subject: dict[str, list[dict]],
    subjects: list[str] | None,
    ntrain: int,
    max_seq_len: int,
    limit_per_subject: int,
    max_subjects: int,
) -> list[EvalExample]:
    subject_names = sorted(test_rows_by_subject) if subjects is None else list(subjects)
    if max_subjects > 0:
        subject_names = subject_names[:max_subjects]

    examples: list[EvalExample] = []
    for subject in subject_names:
        if subject not in test_rows_by_subject:
            raise KeyError(f"unknown subject: {subject}")
        dev_rows = dev_rows_by_subject[subject][:ntrain]
        prefixes = build_fewshot_prefixes(dev_rows, subject)
        subject_rows = test_rows_by_subject[subject]
        if limit_per_subject > 0:
            subject_rows = subject_rows[:limit_per_subject]

        for row_idx, row in enumerate(subject_rows):
            prompt_end = format_example(row["question"], row["choices"], None)
            prompt = ""
            prompt_len = 0
            for fewshot_count in range(len(dev_rows), -1, -1):
                candidate = prefixes[fewshot_count] + prompt_end
                candidate_len = token_length(tokenizer, candidate)
                if candidate_len <= max_seq_len or fewshot_count == 0:
                    prompt = candidate
                    prompt_len = candidate_len
                    break
            examples.append(
                EvalExample(
                    example_id=f"{subject}:{row_idx}",
                    subject=subject,
                    prompt=prompt,
                    answer=int(row["answer"]),
                    prompt_length=prompt_len,
                )
            )
    return examples


def infer_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def batched_accuracy(
    *,
    model,
    tokenizer,
    examples: list[EvalExample],
    batch_size: int,
    choice_token_ids: torch.Tensor,
) -> tuple[float, dict[str, dict[str, float | int]], dict[str, int], dict[str, bool], float]:
    device = infer_input_device(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = sorted(examples, key=lambda item: item.prompt_length)
    per_subject: dict[str, dict[str, float | int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions: dict[str, int] = {}
    correctness: dict[str, bool] = {}

    start_time = time.time()
    last_reported = 0
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        prompts = [item.prompt for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = encoded.to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]
            scores = logits.index_select(dim=-1, index=choice_token_ids.to(device))
            batch_predictions = scores.argmax(dim=-1).cpu().tolist()

        for item, prediction in zip(batch, batch_predictions):
            is_correct = int(prediction) == item.answer
            predictions[item.example_id] = int(prediction)
            correctness[item.example_id] = is_correct
            stats = per_subject[item.subject]
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1

        done = min(offset + batch_size, len(examples))
        if done - last_reported >= 64 or done == len(examples):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"processed {done}/{len(examples)} prompts ({rate:.2f} prompts/s)", flush=True)
            last_reported = done

    runtime_sec = time.time() - start_time
    total_correct = sum(int(stats["correct"]) for stats in per_subject.values())
    total_count = sum(int(stats["total"]) for stats in per_subject.values())
    for subject, stats in per_subject.items():
        total = int(stats["total"])
        stats["accuracy"] = float(stats["correct"]) / total if total else 0.0
    accuracy = float(total_correct) / total_count if total_count else 0.0
    return accuracy, dict(per_subject), predictions, correctness, runtime_sec


def load_rows_by_subject(split: str) -> dict[str, list[dict]]:
    dataset = load_dataset("cais/mmlu", "all", split=split)
    rows_by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        rows_by_subject[row["subject"]].append(row)
    return dict(rows_by_subject)


def parse_dtype(dtype_name: str):
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def load_model_and_tokenizer(
    *,
    model_path: str,
    dtype_name: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": trust_remote_code}
    torch_dtype = parse_dtype(dtype_name)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def evaluate_single_model(
    *,
    label: str,
    model_path: str,
    dtype_name: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    test_rows_by_subject: dict[str, list[dict]],
    dev_rows_by_subject: dict[str, list[dict]],
    subjects: list[str] | None,
    ntrain: int,
    max_seq_len: int,
    limit_per_subject: int,
    max_subjects: int,
    batch_size: int,
) -> tuple[EvalResult, list[EvalExample]]:
    print(f"loading {label} model from {model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    choice_token_ids = []
    for token in CHOICE_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"choice token {token!r} is not a single token for {label}: {token_ids}")
        choice_token_ids.append(token_ids[0])

    examples = build_eval_examples(
        tokenizer=tokenizer,
        test_rows_by_subject=test_rows_by_subject,
        dev_rows_by_subject=dev_rows_by_subject,
        subjects=subjects,
        ntrain=ntrain,
        max_seq_len=max_seq_len,
        limit_per_subject=limit_per_subject,
        max_subjects=max_subjects,
    )
    print(f"{label}: built {len(examples)} prompts", flush=True)

    accuracy, per_subject, predictions, correctness, runtime_sec = batched_accuracy(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        choice_token_ids=torch.tensor(choice_token_ids, dtype=torch.long),
    )

    result = EvalResult(
        label=label,
        model_path=model_path,
        dtype=dtype_name,
        accuracy=accuracy,
        num_examples=len(examples),
        runtime_sec=runtime_sec,
        per_subject=per_subject,
        predictions=predictions,
        correctness=correctness,
    )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, examples


def build_pair_summary(
    reference: EvalResult,
    candidate: EvalResult,
    examples: list[EvalExample],
) -> dict:
    example_ids = [item.example_id for item in examples]
    both_correct = 0
    both_wrong = 0
    ref_only_correct = 0
    cand_only_correct = 0
    per_subject = defaultdict(
        lambda: {
            "both_correct": 0,
            "both_wrong": 0,
            "reference_only_correct": 0,
            "candidate_only_correct": 0,
            "total": 0,
        }
    )

    for item in examples:
        example_id = item.example_id
        ref_correct = reference.correctness[example_id]
        cand_correct = candidate.correctness[example_id]
        subject_stats = per_subject[item.subject]
        subject_stats["total"] += 1
        if ref_correct and cand_correct:
            both_correct += 1
            subject_stats["both_correct"] += 1
        elif (not ref_correct) and (not cand_correct):
            both_wrong += 1
            subject_stats["both_wrong"] += 1
        elif ref_correct and (not cand_correct):
            ref_only_correct += 1
            subject_stats["reference_only_correct"] += 1
        else:
            cand_only_correct += 1
            subject_stats["candidate_only_correct"] += 1

    for subject, stats in per_subject.items():
        total = int(stats["total"])
        stats["reference_win_rate"] = (
            float(stats["reference_only_correct"]) / total if total else 0.0
        )
        stats["candidate_win_rate"] = (
            float(stats["candidate_only_correct"]) / total if total else 0.0
        )

    return {
        "num_examples_compared": len(example_ids),
        "reference_label": reference.label,
        "candidate_label": candidate.label,
        "reference_accuracy": reference.accuracy,
        "candidate_accuracy": candidate.accuracy,
        "delta_accuracy": candidate.accuracy - reference.accuracy,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "reference_only_correct": ref_only_correct,
        "candidate_only_correct": cand_only_correct,
        "per_subject": dict(per_subject),
    }


def result_to_json_dict(
    result: EvalResult,
    *,
    batch_size: int,
    ntrain: int,
    max_seq_len: int,
    limit_per_subject: int,
    max_subjects: int,
    subjects: list[str] | None,
    save_predictions: bool,
) -> dict:
    payload = {
        "label": result.label,
        "model_path": result.model_path,
        "dtype": result.dtype,
        "accuracy": result.accuracy,
        "num_subjects": len(result.per_subject),
        "num_examples": result.num_examples,
        "runtime_sec": result.runtime_sec,
        "batch_size": batch_size,
        "ntrain": ntrain,
        "max_seq_len": max_seq_len,
        "limit_per_subject": limit_per_subject,
        "max_subjects": max_subjects,
        "subjects": subjects,
        "per_subject": result.per_subject,
    }
    if save_predictions:
        payload["predictions"] = result.predictions
        payload["correctness"] = result.correctness
    return payload


def write_summary_markdown(
    *,
    output_path: Path,
    fp_result: EvalResult,
    nvfp4_result: EvalResult,
    pair_summary: dict,
) -> None:
    lines = [
        "# MMLU Comparison",
        "",
        "| Model | Accuracy | Examples | Runtime (s) |",
        "| --- | ---: | ---: | ---: |",
        f"| {fp_result.label} | {fp_result.accuracy:.4f} | {fp_result.num_examples} | {fp_result.runtime_sec:.1f} |",
        f"| {nvfp4_result.label} | {nvfp4_result.accuracy:.4f} | {nvfp4_result.num_examples} | {nvfp4_result.runtime_sec:.1f} |",
        "",
        f"- Delta accuracy (`{nvfp4_result.label}` - `{fp_result.label}`): {pair_summary['delta_accuracy']:.4f}",
        f"- `{fp_result.label}` only correct: {pair_summary['reference_only_correct']}",
        f"- `{nvfp4_result.label}` only correct: {pair_summary['candidate_only_correct']}",
        f"- Both correct: {pair_summary['both_correct']}",
        f"- Both wrong: {pair_summary['both_wrong']}",
        "",
    ]
    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-model-path", required=True, help="Full-precision HF model directory")
    parser.add_argument("--nvfp4-model-path", required=True, help="NVFP4 HF model directory")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and markdown outputs")
    parser.add_argument("--fp-label", default="fp16", help="Label used in reports for the full model")
    parser.add_argument("--nvfp4-label", default="nvfp4", help="Label used in reports for the NVFP4 model")
    parser.add_argument(
        "--fp-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Load dtype for the full-precision model",
    )
    parser.add_argument(
        "--nvfp4-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Load dtype for the NVFP4 model",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--limit-per-subject",
        type=int,
        default=0,
        help="0 means full test split for every subject",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="0 means all subjects",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject list. Overrides alphabetical traversal.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional transformers attn_implementation override",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-example predictions in the per-model JSON files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_rows_by_subject = load_rows_by_subject("dev")
    test_rows_by_subject = load_rows_by_subject("test")
    selected_subjects = [item.strip() for item in args.subjects.split(",") if item.strip()] or None

    fp_result, fp_examples = evaluate_single_model(
        label=args.fp_label,
        model_path=args.fp_model_path,
        dtype_name=args.fp_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        test_rows_by_subject=test_rows_by_subject,
        dev_rows_by_subject=dev_rows_by_subject,
        subjects=selected_subjects,
        ntrain=args.ntrain,
        max_seq_len=args.max_seq_len,
        limit_per_subject=args.limit_per_subject,
        max_subjects=args.max_subjects,
        batch_size=args.batch_size,
    )

    nvfp4_result, nvfp4_examples = evaluate_single_model(
        label=args.nvfp4_label,
        model_path=args.nvfp4_model_path,
        dtype_name=args.nvfp4_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        test_rows_by_subject=test_rows_by_subject,
        dev_rows_by_subject=dev_rows_by_subject,
        subjects=selected_subjects,
        ntrain=args.ntrain,
        max_seq_len=args.max_seq_len,
        limit_per_subject=args.limit_per_subject,
        max_subjects=args.max_subjects,
        batch_size=args.batch_size,
    )

    if [item.example_id for item in fp_examples] != [item.example_id for item in nvfp4_examples]:
        raise RuntimeError("The evaluated question set differs between the two models.")

    pair_summary = build_pair_summary(fp_result, nvfp4_result, fp_examples)

    fp_json = result_to_json_dict(
        fp_result,
        batch_size=args.batch_size,
        ntrain=args.ntrain,
        max_seq_len=args.max_seq_len,
        limit_per_subject=args.limit_per_subject,
        max_subjects=args.max_subjects,
        subjects=selected_subjects,
        save_predictions=args.save_predictions,
    )
    nvfp4_json = result_to_json_dict(
        nvfp4_result,
        batch_size=args.batch_size,
        ntrain=args.ntrain,
        max_seq_len=args.max_seq_len,
        limit_per_subject=args.limit_per_subject,
        max_subjects=args.max_subjects,
        subjects=selected_subjects,
        save_predictions=args.save_predictions,
    )
    summary_json = {
        "fp_result": fp_json,
        "nvfp4_result": nvfp4_json,
        "pair_summary": pair_summary,
    }

    (output_dir / f"{args.fp_label}.json").write_text(json.dumps(fp_json, indent=2))
    (output_dir / f"{args.nvfp4_label}.json").write_text(json.dumps(nvfp4_json, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        fp_result=fp_result,
        nvfp4_result=nvfp4_result,
        pair_summary=pair_summary,
    )

    print(
        json.dumps(
            {
                "fp_label": fp_result.label,
                "fp_accuracy": fp_result.accuracy,
                "nvfp4_label": nvfp4_result.label,
                "nvfp4_accuracy": nvfp4_result.accuracy,
                "delta_accuracy": pair_summary["delta_accuracy"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"saved reports to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
