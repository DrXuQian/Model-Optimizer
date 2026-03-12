# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NVFP4 double-scale initialization followed by repo-style FP8 MSE sweep."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from experimental.nvfp4_entquant.optimizer import absmax_actual_scales
from experimental.nvfp4_scale_inflation.scale_inflation import (
    BLOCK_SIZE,
    FP4_MAX_SCALE_VALUE,
    FP4_VALUES,
    build_layer_metrics,
    encode_actual_scales,
    fp4_codes_to_values,
    list_nvfp4_weight_prefixes,
    quantize_to_fp4_codes,
    quantize_to_nvfp4,
)

_EPS = 1e-6
_FP4_VALUES_F32 = FP4_VALUES.to(torch.float32)
_CODE_SIGNS = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1])
DEFAULT_PRESET = "compress10"


@dataclass(frozen=True)
class DoubleScaleRepoMSESweepConfig:
    alpha: float = 2.0
    min_scale_ratio: float = 1.0
    soft_entropy_budget_ratio: float | None = 0.75
    soft_temperature: float = 0.20
    candidate_chunk_size: int = 32
    block_chunk_size: int = 65536


@dataclass(frozen=True)
class DoubleScaleRepoMSESweepResult:
    packed: torch.Tensor
    dequantized: torch.Tensor
    codes: torch.Tensor
    actual_scales: torch.Tensor
    encoded_scale: torch.Tensor
    encoded_global_scale: torch.Tensor


def preset_config(name: str) -> DoubleScaleRepoMSESweepConfig:
    preset = name.strip().lower()
    if preset == "precision":
        return DoubleScaleRepoMSESweepConfig(
            alpha=2.0,
            min_scale_ratio=1.0,
            soft_entropy_budget_ratio=0.75,
            soft_temperature=0.20,
            candidate_chunk_size=32,
            block_chunk_size=65536,
        )
    if preset == "compress10":
        return DoubleScaleRepoMSESweepConfig(
            alpha=2.0,
            min_scale_ratio=1.0,
            soft_entropy_budget_ratio=0.25,
            soft_temperature=0.20,
            candidate_chunk_size=32,
            block_chunk_size=65536,
        )
    if preset == "mse":
        return DoubleScaleRepoMSESweepConfig(
            alpha=2.0,
            min_scale_ratio=1.0,
            soft_entropy_budget_ratio=None,
            soft_temperature=0.20,
            candidate_chunk_size=32,
            block_chunk_size=65536,
        )
    raise ValueError(f"Unsupported preset: {name}")


def config_with_overrides(
    base: DoubleScaleRepoMSESweepConfig,
    *,
    alpha: float | None = None,
    min_scale_ratio: float | None = None,
    soft_entropy_budget_ratio: float | None = None,
    soft_temperature: float | None = None,
    candidate_chunk_size: int | None = None,
    block_chunk_size: int | None = None,
) -> DoubleScaleRepoMSESweepConfig:
    return DoubleScaleRepoMSESweepConfig(
        alpha=base.alpha if alpha is None else alpha,
        min_scale_ratio=base.min_scale_ratio if min_scale_ratio is None else min_scale_ratio,
        soft_entropy_budget_ratio=(
            base.soft_entropy_budget_ratio if soft_entropy_budget_ratio is None else soft_entropy_budget_ratio
        ),
        soft_temperature=base.soft_temperature if soft_temperature is None else soft_temperature,
        candidate_chunk_size=base.candidate_chunk_size if candidate_chunk_size is None else candidate_chunk_size,
        block_chunk_size=base.block_chunk_size if block_chunk_size is None else block_chunk_size,
    )


class ShardedTensorLoader:
    """Read sharded safetensors tensors by key."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open() as handle:
            self._weight_map = json.load(handle)["weight_map"]
        self._model_dir = model_dir
        self._stack: contextlib.ExitStack | None = None
        self._handles: dict[str, safe_open] = {}

    def __enter__(self) -> "ShardedTensorLoader":
        self._stack = contextlib.ExitStack()
        for shard_file in sorted(set(self._weight_map.values())):
            self._handles[shard_file] = self._stack.enter_context(
                safe_open(str(self._model_dir / shard_file), framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._handles.clear()

    def get_tensor(self, key: str) -> torch.Tensor:
        shard_file = self._weight_map[key]
        return self._handles[shard_file].get_tensor(key)


def valid_fp8_scale_candidates(device: torch.device) -> torch.Tensor:
    """Return all valid positive FP8 E4M3 scale multipliers in [0, 1]."""
    values = torch.arange(0, 128, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float()
    valid = torch.isfinite(values) & (values > 0)
    return values[valid] / FP4_MAX_SCALE_VALUE


def legalize_scales_with_fixed_global(
    scales: torch.Tensor,
    global_max_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode scales using a fixed global maximum, matching repo FP8 sweep semantics."""
    scales = scales.to(torch.float32)
    if isinstance(global_max_scale, torch.Tensor):
        global_max = float(global_max_scale.detach().float().item())
    else:
        global_max = float(global_max_scale)
    if global_max <= 0.0:
        raise ValueError(f"global_max_scale must be positive, got {global_max}")

    global_scale = torch.tensor([FP4_MAX_SCALE_VALUE / global_max], dtype=torch.float32, device=scales.device)
    normalized = (scales * global_scale).clamp(min=0.0, max=FP4_MAX_SCALE_VALUE)
    encoded_scale = normalized.to(torch.float8_e4m3fn)
    actual_scales = encoded_scale.to(torch.float32) / global_scale
    return encoded_scale, global_scale, actual_scales


def _soft_code_logits(scaled: torch.Tensor, temperature: float) -> torch.Tensor:
    codebook = _FP4_VALUES_F32.to(device=scaled.device)
    code_signs = _CODE_SIGNS.to(device=scaled.device, dtype=torch.float32)
    scaled_expanded = scaled.unsqueeze(-1)
    logits = -(scaled_expanded - codebook).abs() / max(float(temperature), _EPS)

    sign = torch.sign(scaled_expanded)
    mismatch = (sign * code_signs < 0).to(logits.dtype)
    logits = logits - 0.25 * mismatch
    logits[..., 0] = logits[..., 0] - 0.10 * (scaled < 0).to(logits.dtype)
    logits[..., 8] = logits[..., 8] - 0.10 * (scaled >= 0).to(logits.dtype)
    return logits


def soft_code_entropy_bits_per_block(scaled_blocks: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    """Return per-block soft-code entropy for [..., block_size] scaled weights."""
    probs = torch.softmax(_soft_code_logits(scaled_blocks, temperature), dim=-1)
    hist = probs.mean(dim=-2)
    return -(hist * torch.log2(hist.clamp_min(_EPS))).sum(dim=-1)


def soft_code_entropy_bits_per_block_chunked(
    scaled_blocks: torch.Tensor,
    *,
    temperature: float = 0.20,
    block_chunk_size: int = 65536,
) -> torch.Tensor:
    """Compute per-block soft-code entropy in chunks along the block dimension."""
    flat = scaled_blocks.reshape(-1, BLOCK_SIZE)
    outputs = []
    for start in range(0, flat.shape[0], block_chunk_size):
        chunk = flat[start : start + block_chunk_size]
        outputs.append(soft_code_entropy_bits_per_block(chunk, temperature=temperature))
    return torch.cat(outputs, dim=0).reshape(*scaled_blocks.shape[:-1])


def mse_sweep_scales_from_fixed_global(
    weight: torch.Tensor,
    global_max_scale: torch.Tensor | float,
    min_scales: torch.Tensor | None = None,
    soft_entropy_max: torch.Tensor | None = None,
    soft_temperature: float = 0.20,
    candidate_chunk_size: int = 16,
    block_chunk_size: int = 65536,
) -> torch.Tensor:
    """Select the best legal FP8 block scale under a fixed global max using MSE."""
    weight = weight.to(torch.float32)
    blocks = weight.reshape(-1, BLOCK_SIZE)
    device = blocks.device
    if isinstance(global_max_scale, torch.Tensor):
        global_max = float(global_max_scale.detach().float().item())
    else:
        global_max = float(global_max_scale)
    candidates = valid_fp8_scale_candidates(device) * global_max
    min_scales_flat = None if min_scales is None else min_scales.to(device=device, dtype=torch.float32).reshape(-1)
    soft_entropy_max_flat = (
        None if soft_entropy_max is None else soft_entropy_max.to(device=device, dtype=torch.float32).reshape(-1)
    )

    best_loss = torch.full((blocks.shape[0],), float("inf"), dtype=torch.float32, device=device)
    best_scale = torch.full((blocks.shape[0],), candidates[0], dtype=torch.float32, device=device)

    for block_start in range(0, blocks.shape[0], block_chunk_size):
        block_end = block_start + block_chunk_size
        blocks_chunk = blocks[block_start:block_end]
        best_loss_chunk = best_loss[block_start:block_end]
        best_scale_chunk = best_scale[block_start:block_end]
        min_scales_chunk = None
        if min_scales_flat is not None:
            min_scales_chunk = min_scales_flat[block_start:block_end]
        soft_entropy_max_chunk = None
        if soft_entropy_max_flat is not None:
            soft_entropy_max_chunk = soft_entropy_max_flat[block_start:block_end]

        for start in range(0, candidates.numel(), candidate_chunk_size):
            cand = candidates[start : start + candidate_chunk_size]
            scaled = blocks_chunk.unsqueeze(0) / cand.view(-1, 1, 1)
            codes = quantize_to_fp4_codes(scaled.reshape(-1)).view(
                cand.numel(), blocks_chunk.shape[0], BLOCK_SIZE
            )
            dequantized = fp4_codes_to_values(codes).to(torch.float32) * cand.view(-1, 1, 1)
            mse = ((dequantized - blocks_chunk.unsqueeze(0)) ** 2).mean(dim=-1)
            if min_scales_chunk is not None:
                allowed = cand.view(-1, 1) >= min_scales_chunk.view(1, -1)
                mse = torch.where(allowed, mse, torch.full_like(mse, float("inf")))
            if soft_entropy_max_chunk is not None:
                soft_entropy = soft_code_entropy_bits_per_block_chunked(
                    scaled,
                    temperature=soft_temperature,
                    block_chunk_size=block_chunk_size,
                )
                allowed = soft_entropy <= soft_entropy_max_chunk.view(1, -1)
                mse = torch.where(allowed, mse, torch.full_like(mse, float("inf")))
            chunk_best_loss, chunk_best_idx = mse.min(dim=0)
            update = chunk_best_loss < best_loss_chunk
            best_loss_chunk[update] = chunk_best_loss[update]
            best_scale_chunk[update] = cand[chunk_best_idx[update]]

        best_loss[block_start:block_end] = best_loss_chunk
        best_scale[block_start:block_end] = best_scale_chunk

    return best_scale.view(*weight.shape[:-1], weight.shape[-1] // BLOCK_SIZE)


def quantize_weight_double_scale_repo_mse_sweep(
    weight: torch.Tensor,
    config: DoubleScaleRepoMSESweepConfig | None = None,
) -> DoubleScaleRepoMSESweepResult:
    """Apply alpha-scaled global max followed by repo-style per-block FP8 MSE sweep."""
    config = config or DoubleScaleRepoMSESweepConfig()
    weight = weight.to(torch.float32)
    absmax_scales = absmax_actual_scales(weight)
    target_global_max = absmax_scales.max() * config.alpha
    min_scales = absmax_scales * config.min_scale_ratio
    soft_entropy_max = None
    if config.soft_entropy_budget_ratio is not None:
        _, _, absmax_fixed_scales = legalize_scales_with_fixed_global(absmax_scales, global_max_scale=target_global_max)
        _, _, double_fixed_scales = legalize_scales_with_fixed_global(
            absmax_scales * config.alpha,
            global_max_scale=target_global_max,
        )
        absmax_soft_entropy = soft_code_entropy_bits_per_block_chunked(
            weight.reshape(-1, BLOCK_SIZE) / absmax_fixed_scales.reshape(-1, 1),
            temperature=config.soft_temperature,
            block_chunk_size=config.block_chunk_size,
        )
        double_soft_entropy = soft_code_entropy_bits_per_block_chunked(
            weight.reshape(-1, BLOCK_SIZE) / double_fixed_scales.reshape(-1, 1),
            temperature=config.soft_temperature,
            block_chunk_size=config.block_chunk_size,
        )
        soft_entropy_max = double_soft_entropy + config.soft_entropy_budget_ratio * (
            absmax_soft_entropy - double_soft_entropy
        )
    swept_scales = mse_sweep_scales_from_fixed_global(
        weight,
        global_max_scale=target_global_max,
        min_scales=min_scales,
        soft_entropy_max=soft_entropy_max,
        soft_temperature=config.soft_temperature,
        candidate_chunk_size=config.candidate_chunk_size,
        block_chunk_size=config.block_chunk_size,
    )
    encoded_scale, encoded_global_scale, actual_scales = legalize_scales_with_fixed_global(
        swept_scales,
        global_max_scale=target_global_max,
    )
    packed, dequantized, codes = quantize_to_nvfp4(weight, actual_scales)
    return DoubleScaleRepoMSESweepResult(
        packed=packed,
        dequantized=dequantized,
        codes=codes,
        actual_scales=actual_scales,
        encoded_scale=encoded_scale,
        encoded_global_scale=encoded_global_scale,
    )


def _metrics_from_result(weight: torch.Tensor, result: DoubleScaleRepoMSESweepResult) -> dict[str, float]:
    metrics = build_layer_metrics(
        packed=result.packed,
        codes=result.codes,
        dequantized_weight=result.dequantized,
        baseline_weight=weight,
        scales=result.actual_scales,
        reference_scales=result.actual_scales,
        global_scale=result.encoded_global_scale,
    )
    return {
        "mean_entropy_bits": metrics.mean_entropy_bits,
        "compression_rate": metrics.compression_rate,
        "mse": metrics.mse_to_baseline,
    }


def _quantize_from_scales(weight: torch.Tensor, scales: torch.Tensor) -> DoubleScaleRepoMSESweepResult:
    encoded_scale, encoded_global_scale = encode_actual_scales(scales)
    actual_scales = encoded_scale.to(torch.float32) / encoded_global_scale
    packed, dequantized, codes = quantize_to_nvfp4(weight, actual_scales)
    return DoubleScaleRepoMSESweepResult(
        packed=packed,
        dequantized=dequantized,
        codes=codes,
        actual_scales=actual_scales,
        encoded_scale=encoded_scale,
        encoded_global_scale=encoded_global_scale,
    )


def evaluate_layer(
    prefix: str,
    *,
    full_precision_loader: ShardedTensorLoader,
    row_limit: int,
    config: DoubleScaleRepoMSESweepConfig,
) -> dict[str, Any]:
    weight = full_precision_loader.get_tensor(prefix + ".weight").to(torch.float32)
    if row_limit > 0:
        if weight.ndim != 2:
            raise ValueError(f"row_limit only supports 2D weights, got {tuple(weight.shape)}")
        weight = weight[:row_limit].contiguous()

    absmax_scales = absmax_actual_scales(weight)
    absmax_result = _quantize_from_scales(weight, absmax_scales)
    double_result = _quantize_from_scales(weight, absmax_scales * config.alpha)
    swept_result = quantize_weight_double_scale_repo_mse_sweep(weight, config=config)

    report = {
        "layer": prefix,
        "shape": list(weight.shape),
        "num_weights": int(weight.numel()),
        "absmax": _metrics_from_result(weight, absmax_result),
        "double_scale": _metrics_from_result(weight, double_result),
        "double_scale_repo_mse_sweep": _metrics_from_result(weight, swept_result),
        "alpha": config.alpha,
        "min_scale_ratio": config.min_scale_ratio,
        "soft_entropy_budget_ratio": config.soft_entropy_budget_ratio,
    }
    print(
        f"{prefix[-64:]:<64} "
        f"abs={report['absmax']['compression_rate'] * 100:5.1f}% "
        f"double={report['double_scale']['compression_rate'] * 100:5.1f}% "
        f"sweep={report['double_scale_repo_mse_sweep']['compression_rate'] * 100:5.1f}% "
        f"sweep_mse={report['double_scale_repo_mse_sweep']['mse']:.3e}"
    )
    return report


def _weighted_average(results: list[dict[str, Any]], key: str, field: str) -> float | None:
    total_weight = 0
    total = 0.0
    for result in results:
        item = result.get(key)
        if item is None:
            continue
        num_weights = int(result["num_weights"])
        total_weight += num_weights
        total += num_weights * float(item[field])
    if total_weight == 0:
        return None
    return total / total_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument("--template-nvfp4-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=("precision", "compress10", "mse"),
        default=DEFAULT_PRESET,
        help="Default script mode. 'compress10' is the default.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-scale-ratio", type=float, default=None)
    parser.add_argument("--soft-entropy-budget-ratio", type=float, default=None)
    parser.add_argument("--soft-temperature", type=float, default=None)
    parser.add_argument("--candidate-chunk-size", type=int, default=None)
    parser.add_argument("--block-chunk-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = Path(args.template_nvfp4_dir) / "model.safetensors"
    prefixes = list_nvfp4_weight_prefixes(
        template_path,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )
    if args.max_layers > 0:
        prefixes = prefixes[: args.max_layers]

    config = config_with_overrides(
        preset_config(args.preset),
        alpha=args.alpha,
        min_scale_ratio=args.min_scale_ratio,
        soft_entropy_budget_ratio=args.soft_entropy_budget_ratio,
        soft_temperature=args.soft_temperature,
        candidate_chunk_size=args.candidate_chunk_size,
        block_chunk_size=args.block_chunk_size,
    )

    results = []
    with ShardedTensorLoader(args.full_precision_model_dir) as loader:
        for prefix in prefixes:
            results.append(
                evaluate_layer(
                    prefix,
                    full_precision_loader=loader,
                    row_limit=args.row_limit,
                    config=config,
                )
            )

    summary = {
        "full_precision_model_dir": args.full_precision_model_dir,
        "template_nvfp4_dir": args.template_nvfp4_dir,
        "config": asdict(config),
        "aggregate": {
            "absmax_weighted_compression_rate": _weighted_average(results, "absmax", "compression_rate"),
            "absmax_weighted_mse": _weighted_average(results, "absmax", "mse"),
            "double_scale_weighted_compression_rate": _weighted_average(results, "double_scale", "compression_rate"),
            "double_scale_weighted_mse": _weighted_average(results, "double_scale", "mse"),
            "double_scale_repo_mse_sweep_weighted_compression_rate": _weighted_average(
                results, "double_scale_repo_mse_sweep", "compression_rate"
            ),
            "double_scale_repo_mse_sweep_weighted_mse": _weighted_average(
                results, "double_scale_repo_mse_sweep", "mse"
            ),
        },
        "layers": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
