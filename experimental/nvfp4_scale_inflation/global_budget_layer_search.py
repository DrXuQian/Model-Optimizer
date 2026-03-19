# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer global entropy-budget search for NVFP4 double-scale MSE sweep."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import (
    BLOCK_SIZE,
    DoubleScaleRepoMSESweepConfig,
    ShardedTensorLoader,
    _metrics_from_result,
    _quantize_from_scales,
    absmax_actual_scales,
    config_with_overrides,
    hard_code_entropy_bits_per_block_chunked,
    preset_config,
    quantize_weight_double_scale_repo_mse_sweep,
    soft_code_entropy_bits_per_block_chunked,
    valid_fp8_scale_candidates,
)
from experimental.nvfp4_scale_inflation.scale_inflation import fp4_codes_to_values, quantize_to_fp4_codes

_INF = float("inf")


def _pareto_mask(mse_table: torch.Tensor, entropy_table: torch.Tensor, valid_table: torch.Tensor) -> torch.Tensor:
    """Mask dominated candidates for each block."""
    num_blocks, num_candidates = mse_table.shape
    mask = torch.zeros_like(valid_table)
    mse_cpu = mse_table.cpu()
    entropy_cpu = entropy_table.cpu()
    valid_cpu = valid_table.cpu()

    for block_idx in range(num_blocks):
        valid_idx = torch.nonzero(valid_cpu[block_idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        ent = entropy_cpu[block_idx, valid_idx]
        mse = mse_cpu[block_idx, valid_idx]
        order = torch.argsort(ent, stable=True)
        best_mse = _INF
        keep = torch.zeros_like(valid_idx, dtype=torch.bool)
        for pos in order.tolist():
            candidate_mse = float(mse[pos].item())
            if candidate_mse < best_mse - 1e-12:
                keep[pos] = True
                best_mse = candidate_mse
        mask[block_idx, valid_idx[keep]] = True
    return mask.to(valid_table.device)


def _select_by_lambda(
    *,
    candidates: torch.Tensor,
    mse_table: torch.Tensor,
    entropy_table: torch.Tensor,
    valid_mask: torch.Tensor,
    lam: float,
) -> dict[str, Any]:
    objective = mse_table + float(lam) * entropy_table
    objective = torch.where(valid_mask, objective, torch.full_like(objective, _INF))
    selected_idx = torch.argmin(objective, dim=1)
    row_idx = torch.arange(selected_idx.shape[0], device=selected_idx.device)
    selected_scales = candidates[selected_idx]
    selected_mse = mse_table[row_idx, selected_idx]
    selected_entropy = entropy_table[row_idx, selected_idx]
    return {
        "lambda": float(lam),
        "indices": selected_idx,
        "scales": selected_scales,
        "total_entropy": float(selected_entropy.sum().item()),
        "mean_entropy": float(selected_entropy.mean().item()),
        "mean_mse": float(selected_mse.mean().item()),
    }


def _soft_total_entropy(weight: torch.Tensor, scales: torch.Tensor, *, temperature: float, block_chunk_size: int) -> float:
    blocks = weight.reshape(-1, BLOCK_SIZE)
    scaled = blocks / scales.reshape(-1, 1)
    return float(
        soft_code_entropy_bits_per_block_chunked(
            scaled,
            temperature=temperature,
            block_chunk_size=block_chunk_size,
        ).sum().item()
    )


def _total_entropy(
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    temperature: float,
    block_chunk_size: int,
    entropy_proxy: str = "soft",
) -> float:
    blocks = weight.reshape(-1, BLOCK_SIZE)
    scaled = blocks / scales.reshape(-1, 1)
    if entropy_proxy == "hard":
        return float(
            hard_code_entropy_bits_per_block_chunked(
                scaled,
                block_chunk_size=block_chunk_size,
            ).sum().item()
        )
    return float(
        soft_code_entropy_bits_per_block_chunked(
            scaled,
            temperature=temperature,
            block_chunk_size=block_chunk_size,
        ).sum().item()
    )


def _global_budget_search(
    weight: torch.Tensor,
    *,
    config: DoubleScaleRepoMSESweepConfig,
    target_total_entropy: float,
    target_hard_compression_rate: float,
    hard_gap_tolerance: float = 0.01,
    include_tensors: bool = False,
    entropy_proxy: str = "soft",
) -> dict[str, Any]:
    weight = weight.to(torch.float32)
    blocks = weight.reshape(-1, BLOCK_SIZE)
    absmax_scale_grid = absmax_actual_scales(weight)
    absmax_scales = absmax_scale_grid.reshape(-1)
    global_max = float(absmax_scales.max().item() * config.alpha)
    candidates = valid_fp8_scale_candidates(weight.device) * global_max
    min_scales = absmax_scales * config.min_scale_ratio

    num_blocks = blocks.shape[0]
    num_candidates = candidates.numel()
    mse_table = torch.empty((num_blocks, num_candidates), dtype=torch.float32, device=weight.device)
    entropy_table = torch.empty((num_blocks, num_candidates), dtype=torch.float32, device=weight.device)
    valid_table = torch.empty((num_blocks, num_candidates), dtype=torch.bool, device=weight.device)

    with torch.inference_mode():
        for start in range(0, num_candidates, config.candidate_chunk_size):
            cand = candidates[start : start + config.candidate_chunk_size]
            scaled = blocks.unsqueeze(0) / cand.view(-1, 1, 1)
            quantized_codes = quantize_to_fp4_codes(scaled.reshape(-1)).view(cand.numel(), num_blocks, BLOCK_SIZE)
            dequantized = fp4_codes_to_values(quantized_codes).to(torch.float32) * cand.view(-1, 1, 1)
            mse_chunk = ((dequantized - blocks.unsqueeze(0)) ** 2).mean(dim=-1).transpose(0, 1)
            if entropy_proxy == "hard":
                entropy_chunk = hard_code_entropy_bits_per_block_chunked(
                    scaled,
                    block_chunk_size=config.block_chunk_size,
                ).transpose(0, 1)
            else:
                entropy_chunk = soft_code_entropy_bits_per_block_chunked(
                    scaled,
                    temperature=config.soft_temperature,
                    block_chunk_size=config.block_chunk_size,
                ).transpose(0, 1)
            valid_chunk = cand.view(1, -1) >= min_scales.view(-1, 1)
            mse_table[:, start : start + cand.numel()] = mse_chunk
            entropy_table[:, start : start + cand.numel()] = entropy_chunk
            valid_table[:, start : start + cand.numel()] = valid_chunk

    pareto_mask = _pareto_mask(mse_table, entropy_table, valid_table)

    lambda_values = [0.0]
    lambda_hi = 1.0
    probe = _select_by_lambda(
        candidates=candidates,
        mse_table=mse_table,
        entropy_table=entropy_table,
        valid_mask=pareto_mask,
        lam=lambda_hi,
    )
    while probe["total_entropy"] > target_total_entropy and lambda_hi < 1e6:
        lambda_values.append(lambda_hi)
        lambda_hi *= 2.0
        probe = _select_by_lambda(
            candidates=candidates,
            mse_table=mse_table,
            entropy_table=entropy_table,
            valid_mask=pareto_mask,
            lam=lambda_hi,
        )
    lambda_values.extend(torch.logspace(-6, torch.log10(torch.tensor(lambda_hi)).item(), steps=48).tolist())

    candidates_by_lambda: list[dict[str, Any]] = []
    best_feasible_soft: dict[str, Any] | None = None
    best_closest_soft: dict[str, Any] | None = None

    for lam in sorted(set(float(x) for x in lambda_values)):
        selection = _select_by_lambda(
            candidates=candidates,
            mse_table=mse_table,
            entropy_table=entropy_table,
            valid_mask=pareto_mask,
            lam=lam,
        )
        entropy_gap = selection["total_entropy"] - target_total_entropy
        selection["entropy_gap"] = float(entropy_gap)
        candidates_by_lambda.append(selection)
        if entropy_gap <= 1e-9:
            if best_feasible_soft is None or selection["mean_mse"] < best_feasible_soft["mean_mse"] - 1e-12:
                best_feasible_soft = selection
        abs_gap = abs(entropy_gap)
        if best_closest_soft is None or abs_gap < abs(best_closest_soft["entropy_gap"]) - 1e-9:
            best_closest_soft = selection

    unique_candidates: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for selection in candidates_by_lambda:
        key = selection["indices"].detach().cpu().numpy().tobytes()
        if key in seen:
            continue
        seen.add(key)
        global_result = _quantize_from_scales(weight, selection["scales"].reshape_as(absmax_scale_grid))
        hard_metrics = _metrics_from_result(weight, global_result)
        unique_candidates.append(
            {
                "lambda": selection["lambda"],
                "total_soft_entropy": selection["total_entropy"],
                "mean_soft_entropy": selection["mean_entropy"],
                "mean_candidate_mse": selection["mean_mse"],
                "entropy_gap": selection["entropy_gap"],
                "hard_metrics": hard_metrics,
                "scales": selection["scales"],
            }
        )

    best_feasible_hard: dict[str, Any] | None = None
    best_closest_hard: dict[str, Any] | None = None
    best_within_tolerance: dict[str, Any] | None = None
    for item in unique_candidates:
        hard_gap = item["hard_metrics"]["compression_rate"] - target_hard_compression_rate
        item["hard_compression_gap"] = hard_gap
        if abs(hard_gap) <= hard_gap_tolerance:
            if best_within_tolerance is None or item["hard_metrics"]["mse"] < best_within_tolerance["hard_metrics"]["mse"] - 1e-12:
                best_within_tolerance = item
        if hard_gap >= -1e-9:
            if best_feasible_hard is None or item["hard_metrics"]["mse"] < best_feasible_hard["hard_metrics"]["mse"] - 1e-12:
                best_feasible_hard = item
        abs_gap = abs(hard_gap)
        if best_closest_hard is None:
            best_closest_hard = item
        else:
            best_abs_gap = abs(best_closest_hard["hard_compression_gap"])
            if abs_gap < best_abs_gap - 1e-9 or (
                abs_gap <= best_abs_gap + 1e-9
                and item["hard_metrics"]["mse"] < best_closest_hard["hard_metrics"]["mse"] - 1e-12
            ):
                best_closest_hard = item

    chosen = best_within_tolerance or best_closest_hard or best_feasible_hard
    assert chosen is not None

    payload = {
        "target_total_soft_entropy": float(target_total_entropy),
        "target_hard_compression_rate": float(target_hard_compression_rate),
        "hard_gap_tolerance": float(hard_gap_tolerance),
        "selected_lambda": chosen["lambda"],
        "selected_total_soft_entropy": chosen["total_soft_entropy"],
        "selected_mean_soft_entropy": chosen["mean_soft_entropy"],
        "selected_mean_candidate_mse": chosen["mean_candidate_mse"],
        "selected_hard_compression_gap": float(chosen["hard_compression_gap"]),
        "selected_within_hard_tolerance": abs(chosen["hard_compression_gap"]) <= hard_gap_tolerance,
        "pareto_mean_candidates_per_block": float(pareto_mask.sum(dim=1).float().mean().item()),
        "hard_metrics": chosen["hard_metrics"],
        "num_unique_lambda_solutions": len(unique_candidates),
        "num_blocks": int(num_blocks),
        "num_candidates": int(num_candidates),
    }
    if include_tensors:
        payload["selected_scales"] = chosen["scales"]
    return payload


def quantize_weight_global_budget_repo_mse_sweep(
    weight: torch.Tensor,
    *,
    config: DoubleScaleRepoMSESweepConfig,
    hard_gap_tolerance: float = 0.01,
    entropy_proxy: str = "soft",
) -> dict[str, Any]:
    weight = weight.to(torch.float32)
    absmax_scales = absmax_actual_scales(weight)
    absmax_result = _quantize_from_scales(weight, absmax_scales)
    double_result = _quantize_from_scales(weight, absmax_scales * config.alpha)
    local_result = quantize_weight_double_scale_repo_mse_sweep(weight, config=config)

    absmax_metrics = _metrics_from_result(weight, absmax_result)
    double_metrics = _metrics_from_result(weight, double_result)
    local_metrics = _metrics_from_result(weight, local_result)
    local_total_entropy = _total_entropy(
        weight,
        local_result.actual_scales,
        temperature=config.soft_temperature,
        block_chunk_size=config.block_chunk_size,
        entropy_proxy=entropy_proxy,
    )
    global_search = _global_budget_search(
        weight,
        config=config,
        target_total_entropy=local_total_entropy,
        target_hard_compression_rate=local_metrics["compression_rate"],
        hard_gap_tolerance=hard_gap_tolerance,
        include_tensors=True,
        entropy_proxy=entropy_proxy,
    )
    global_result = _quantize_from_scales(weight, global_search["selected_scales"].reshape_as(absmax_scales))
    global_metrics = _metrics_from_result(weight, global_result)

    report = {
        "absmax": absmax_metrics,
        "double_scale": double_metrics,
        "local_budget": {
            **local_metrics,
            "total_soft_entropy": local_total_entropy,
        },
        "global_budget": {
            key: value for key, value in global_search.items() if key != "selected_scales"
        },
    }
    report["global_budget"]["hard_metrics"] = global_metrics
    return {
        "absmax_result": absmax_result,
        "double_result": double_result,
        "local_result": local_result,
        "global_result": global_result,
        "report": report,
    }


def run_layer_global_budget_search(
    *,
    full_precision_model_dir: str | Path,
    layer: str,
    config: DoubleScaleRepoMSESweepConfig,
    row_limit: int = 128,
    entropy_proxy: str = "soft",
) -> dict[str, Any]:
    with ShardedTensorLoader(full_precision_model_dir) as loader:
        weight = loader.get_tensor(layer + ".weight").to(torch.float32)
    if row_limit > 0:
        if weight.ndim != 2:
            raise ValueError(f"row_limit only supports 2D weights, got {tuple(weight.shape)}")
        weight = weight[:row_limit].contiguous()

    quantized = quantize_weight_global_budget_repo_mse_sweep(
        weight,
        config=config,
        entropy_proxy=entropy_proxy,
    )
    absmax_metrics = quantized["report"]["absmax"]
    double_metrics = quantized["report"]["double_scale"]
    local_metrics = quantized["report"]["local_budget"]
    global_search = quantized["report"]["global_budget"]

    payload = {
        "layer": layer,
        "shape": list(weight.shape),
        "num_weights": int(weight.numel()),
        "config": asdict(config),
        "row_limit": int(row_limit),
        "absmax": absmax_metrics,
        "double_scale": double_metrics,
        "local_budget": local_metrics,
        "global_budget": global_search,
    }

    print(
        f"{layer} rows={weight.shape[0]} "
        f"abs={absmax_metrics['compression_rate'] * 100:.2f}%/{absmax_metrics['mse']:.3e} "
        f"double={double_metrics['compression_rate'] * 100:.2f}%/{double_metrics['mse']:.3e} "
        f"local={local_metrics['compression_rate'] * 100:.2f}%/{local_metrics['mse']:.3e} "
        f"global={global_search['hard_metrics']['compression_rate'] * 100:.2f}%/"
        f"{global_search['hard_metrics']['mse']:.3e}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument(
        "--layer",
        default="model.layers.0.self_attn.q_proj",
        help="Weight prefix without the trailing .weight",
    )
    parser.add_argument("--row-limit", type=int, default=128)
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
    parser.add_argument(
        "--entropy-proxy",
        choices=("soft", "hard"),
        default="soft",
        help="Entropy proxy for search: 'soft' (default) or 'hard' (frequency-based).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_with_overrides(
        preset_config(args.preset),
        alpha=args.alpha,
        min_scale_ratio=args.min_scale_ratio,
        soft_entropy_budget_ratio=args.soft_entropy_budget_ratio,
        soft_temperature=args.soft_temperature,
        candidate_chunk_size=args.candidate_chunk_size,
        block_chunk_size=args.block_chunk_size,
    )

    payload = run_layer_global_budget_search(
        full_precision_model_dir=args.full_precision_model_dir,
        layer=args.layer,
        config=config,
        row_limit=args.row_limit,
        entropy_proxy=args.entropy_proxy,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
