# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched MMLU evaluation for Hugging Face causal LM checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LABELS = ["A", "B", "C", "D"]
CHOICE_TOKENS = [" A", " B", " C", " D"]
DEFAULT_MAX_SEQ_LEN = 4096


@dataclass
class EvalExample:
    subject: str
    prompt: str
    answer: int
    prompt_length: int


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

        for row in subject_rows:
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
) -> tuple[float, dict[str, dict[str, float | int]]]:
    device = infer_input_device(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = sorted(examples, key=lambda item: item.prompt_length)
    per_subject: dict[str, dict[str, float | int]] = defaultdict(lambda: {"correct": 0, "total": 0})

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
            predictions = scores.argmax(dim=-1).cpu().tolist()

        for item, prediction in zip(batch, predictions):
            stats = per_subject[item.subject]
            stats["total"] += 1
            if int(prediction) == item.answer:
                stats["correct"] += 1

        done = min(offset + batch_size, len(examples))
        if done - last_reported >= 64 or done == len(examples):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"processed {done}/{len(examples)} prompts ({rate:.2f} prompts/s)", flush=True)
            last_reported = done

    total_correct = sum(int(stats["correct"]) for stats in per_subject.values())
    total_count = sum(int(stats["total"]) for stats in per_subject.values())
    for subject, stats in per_subject.items():
        total = int(stats["total"])
        stats["accuracy"] = float(stats["correct"]) / total if total else 0.0
    return (float(total_correct) / total_count if total_count else 0.0), dict(per_subject)


def load_rows_by_subject(split: str) -> dict[str, list[dict]]:
    dataset = load_dataset("cais/mmlu", "all", split=split)
    rows_by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        rows_by_subject[row["subject"]].append(row)
    return dict(rows_by_subject)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument("--output-json", required=True, help="Where to write the report")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="bfloat16",
        help="Model load dtype",
    )
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
        help="Comma-separated explicit subject list. Overrides alphabetical traversal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    choice_token_ids = []
    for token in CHOICE_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"choice token {token!r} is not a single token: {token_ids}")
        choice_token_ids.append(token_ids[0])

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=dtype_map[args.dtype],
    )
    model.eval()

    dev_rows_by_subject = load_rows_by_subject("dev")
    test_rows_by_subject = load_rows_by_subject("test")
    selected_subjects = [item.strip() for item in args.subjects.split(",") if item.strip()] or None
    examples = build_eval_examples(
        tokenizer=tokenizer,
        test_rows_by_subject=test_rows_by_subject,
        dev_rows_by_subject=dev_rows_by_subject,
        subjects=selected_subjects,
        ntrain=args.ntrain,
        max_seq_len=args.max_seq_len,
        limit_per_subject=args.limit_per_subject,
        max_subjects=args.max_subjects,
    )
    print(f"built {len(examples)} prompts", flush=True)

    accuracy, per_subject = batched_accuracy(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=args.batch_size,
        choice_token_ids=torch.tensor(choice_token_ids, dtype=torch.long),
    )

    result = {
        "model_path": args.model_path,
        "accuracy": accuracy,
        "num_subjects": len(per_subject),
        "num_examples": len(examples),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "ntrain": args.ntrain,
        "max_seq_len": args.max_seq_len,
        "limit_per_subject": args.limit_per_subject,
        "max_subjects": args.max_subjects,
        "subjects": selected_subjects,
        "per_subject": per_subject,
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["accuracy", "num_subjects", "num_examples"]}, indent=2), flush=True)
    print(f"saved report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
