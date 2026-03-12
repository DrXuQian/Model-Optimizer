# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run global-budget layer search across multiple layers in parallel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import (
    DoubleScaleRepoMSESweepConfig,
    config_with_overrides,
    preset_config,
)
from experimental.nvfp4_scale_inflation.global_budget_layer_search import run_layer_global_budget_search


def _weighted_average(items: list[dict[str, Any]], key: str, field: str) -> float | None:
    total_weight = 0
    total = 0.0
    for item in items:
        metrics = item.get(key)
        if metrics is None:
            continue
        num_weights = int(item["num_weights"])
        total_weight += num_weights
        total += num_weights * float(metrics[field])
    if total_weight == 0:
        return None
    return total / total_weight


def _run_one(
    *,
    full_precision_model_dir: str,
    layer: str,
    config: DoubleScaleRepoMSESweepConfig,
    row_limit: int,
) -> dict[str, Any]:
    return run_layer_global_budget_search(
        full_precision_model_dir=full_precision_model_dir,
        layer=layer,
        config=config,
        row_limit=row_limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument(
        "--layers",
        default="model.layers.0.self_attn.q_proj,model.layers.0.self_attn.k_proj,model.layers.0.self_attn.v_proj,model.layers.0.self_attn.o_proj",
        help="Comma-separated layer prefixes without trailing .weight",
    )
    parser.add_argument("--row-limit", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--preset",
        choices=("precision", "compress10", "mse"),
        default="compress10",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-scale-ratio", type=float, default=None)
    parser.add_argument("--soft-entropy-budget-ratio", type=float, default=None)
    parser.add_argument("--soft-temperature", type=float, default=None)
    parser.add_argument("--candidate-chunk-size", type=int, default=None)
    parser.add_argument("--block-chunk-size", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = [item.strip() for item in args.layers.split(",") if item.strip()]
    if not layers:
        raise ValueError("No layers specified")

    config = config_with_overrides(
        preset_config(args.preset),
        alpha=args.alpha,
        min_scale_ratio=args.min_scale_ratio,
        soft_entropy_budget_ratio=args.soft_entropy_budget_ratio,
        soft_temperature=args.soft_temperature,
        candidate_chunk_size=args.candidate_chunk_size,
        block_chunk_size=args.block_chunk_size,
    )

    max_workers = max(1, min(int(args.jobs), len(layers), os.cpu_count() or 1))
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                full_precision_model_dir=args.full_precision_model_dir,
                layer=layer,
                config=config,
                row_limit=args.row_limit,
            ): layer
            for layer in layers
        }
        for future in as_completed(futures):
            layer = futures[future]
            result = future.result()
            print(
                f"completed {layer} "
                f"local={result['local_budget']['compression_rate'] * 100:.2f}%/"
                f"{result['local_budget']['mse']:.3e} "
                f"global={result['global_budget']['hard_metrics']['compression_rate'] * 100:.2f}%/"
                f"{result['global_budget']['hard_metrics']['mse']:.3e}",
                flush=True,
            )
            results.append(result)

    results.sort(key=lambda item: item["layer"])
    summary = {
        "full_precision_model_dir": args.full_precision_model_dir,
        "layers_requested": layers,
        "row_limit": int(args.row_limit),
        "jobs": int(max_workers),
        "config": asdict(config),
        "aggregate": {
            "local_weighted_compression_rate": _weighted_average(results, "local_budget", "compression_rate"),
            "local_weighted_mse": _weighted_average(results, "local_budget", "mse"),
            "global_weighted_compression_rate": None,
            "global_weighted_mse": None,
        },
        "layers": results,
    }

    # Compute aggregate fields for nested global hard metrics explicitly.
    total_weight = sum(int(item["num_weights"]) for item in results)
    if total_weight > 0:
        global_comp = sum(
            int(item["num_weights"]) * float(item["global_budget"]["hard_metrics"]["compression_rate"])
            for item in results
        ) / total_weight
        global_mse = sum(
            int(item["num_weights"]) * float(item["global_budget"]["hard_metrics"]["mse"])
            for item in results
        ) / total_weight
        summary["aggregate"]["global_weighted_compression_rate"] = global_comp
        summary["aggregate"]["global_weighted_mse"] = global_mse

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
