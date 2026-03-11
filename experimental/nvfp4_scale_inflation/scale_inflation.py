# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 scale inflation experiment for exported safetensors checkpoints."""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


BLOCK_SIZE = 16
CHUNK_SIZE_BYTES = 32 * 1024
FP4_MAX_SCALE_VALUE = 448.0
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@dataclass
class LayerMetrics:
    mean_entropy_bits: float
    std_entropy_bits: float
    compression_rate: float
    compressed_size_ratio: float
    effective_bits_per_weight: float
    mse_to_baseline: float
    global_scale: float
    scale_ratio_mean: float
    scale_ratio_std: float
    zero_fraction: float
    small_code_fraction: float
    n_chunks: int


def compute_entropy_bits(data: np.ndarray) -> float:
    """Compute Shannon entropy in bits/symbol for a uint8 array."""
    if data.size == 0:
        return 0.0
    counts = np.bincount(data.reshape(-1), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(data.size)
    return float(-(probs * np.log2(probs)).sum())


def compute_nvfp4_compression_rate(
    weight_packed: np.ndarray,
    chunk_size_bytes: int = CHUNK_SIZE_BYTES,
) -> dict[str, Any]:
    """Estimate NVFP4 byte-stream compressibility via chunked entropy."""
    assert weight_packed.dtype == np.uint8, f"expected uint8, got {weight_packed.dtype}"
    flat = weight_packed.reshape(-1)
    if flat.size == 0:
        return {
            "mean_entropy_bits": 0.0,
            "std_entropy_bits": 0.0,
            "compression_rate": 0.0,
            "compressed_size_ratio": 0.0,
            "n_chunks": 0,
            "per_chunk_entropy": [],
        }

    entropies = []
    for start in range(0, flat.size, chunk_size_bytes):
        chunk = flat[start : start + chunk_size_bytes]
        even_bytes = chunk[0::2]
        odd_bytes = chunk[1::2]
        h_even = compute_entropy_bits(even_bytes)
        h_odd = compute_entropy_bits(odd_bytes)
        entropies.append((h_even + h_odd) / 2.0)

    mean_entropy = float(np.mean(entropies))
    std_entropy = float(np.std(entropies))
    compressed_ratio = mean_entropy / 8.0
    return {
        "mean_entropy_bits": mean_entropy,
        "std_entropy_bits": std_entropy,
        "compression_rate": float(1.0 - compressed_ratio),
        "compressed_size_ratio": float(compressed_ratio),
        "n_chunks": len(entropies),
        "per_chunk_entropy": entropies,
    }


def _to_device_constants(device: torch.device) -> torch.Tensor:
    values = FP4_VALUES.to(device=device)
    return values


def unpack_nvfp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed NVFP4 codes into uint8 nibbles."""
    unpacked_shape = list(packed.shape)
    unpacked_shape[-1] *= 2
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8, device=packed.device)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    return unpacked


def pack_nvfp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 NVFP4 codes into uint8 bytes."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {codes.shape[-1]}")
    return (codes[..., 1::2] << 4) | codes[..., 0::2]


def fp4_codes_to_values(codes: torch.Tensor) -> torch.Tensor:
    """Convert NVFP4 codes to E2M1 floating-point values."""
    values = _to_device_constants(codes.device)
    return values[codes.long()]


def quantize_to_fp4_codes(weight: torch.Tensor) -> torch.Tensor:
    """Quantize scaled weights to NVFP4 code indices using compressed-tensors rules."""
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinals = torch.zeros_like(weight_abs, dtype=torch.uint8)
    ordinals[(weight_abs > 0.25) & (weight_abs < 0.75)] = 1
    ordinals[(weight_abs >= 0.75) & (weight_abs <= 1.25)] = 2
    ordinals[(weight_abs > 1.25) & (weight_abs < 1.75)] = 3
    ordinals[(weight_abs >= 1.75) & (weight_abs <= 2.5)] = 4
    ordinals[(weight_abs > 2.5) & (weight_abs < 3.5)] = 5
    ordinals[(weight_abs >= 3.5) & (weight_abs <= 5.0)] = 6
    ordinals[weight_abs > 5.0] = 7
    return (sign_bit << 3) + ordinals


def dequantize_nvfp4_from_checkpoint(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a checkpoint tensor and return weights plus actual per-block scales."""
    if weight_packed.dtype != torch.uint8:
        raise TypeError(f"weight_packed must be uint8, got {weight_packed.dtype}")

    unpacked_codes = unpack_nvfp4_codes(weight_packed)
    unpacked_values = fp4_codes_to_values(unpacked_codes).to(torch.float32)
    actual_scales = weight_scale.to(torch.float32) / weight_global_scale.to(torch.float32)
    if unpacked_values.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"unpacked last dim must be divisible by {BLOCK_SIZE}")

    weight = unpacked_values.view(*unpacked_values.shape[:-1], -1, BLOCK_SIZE)
    weight = weight * actual_scales.unsqueeze(-1)
    return weight.reshape_as(unpacked_values), actual_scales


def encode_actual_scales(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode effective per-block scales back into compressed-tensors NVFP4 format."""
    scales = scales.to(torch.float32)
    max_scale = torch.max(scales)
    if float(max_scale) == 0.0:
        global_scale = torch.tensor([1.0], dtype=torch.float32, device=scales.device)
        fp8_scale = torch.zeros_like(scales, dtype=torch.float8_e4m3fn)
        return fp8_scale, global_scale

    global_scale = (FP4_MAX_SCALE_VALUE / max_scale).to(torch.float32).reshape(1)
    normalized = (scales * global_scale).clamp(min=0.0, max=FP4_MAX_SCALE_VALUE)
    fp8_scale = normalized.to(torch.float8_e4m3fn)
    return fp8_scale, global_scale


def quantize_to_nvfp4(
    weight: torch.Tensor,
    actual_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights with provided actual per-block scales."""
    scaled_weight = weight.view(*weight.shape[:-1], -1, BLOCK_SIZE) / actual_scales.unsqueeze(-1)
    codes = quantize_to_fp4_codes(scaled_weight.reshape_as(weight))
    packed = pack_nvfp4_codes(codes)
    dequantized = fp4_codes_to_values(codes).view(*weight.shape[:-1], -1, BLOCK_SIZE)
    dequantized = dequantized * actual_scales.unsqueeze(-1)
    return packed, dequantized.reshape_as(weight), codes


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(y.dtype).eps
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))


def optimize_scales_ste(
    weight: torch.Tensor,
    original_scales: torch.Tensor,
    initial_scales: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    block_chunk_size: int = 131072,
) -> torch.Tensor:
    """Optimize per-block scales with Adam and a straight-through estimator."""
    if steps <= 0:
        return initial_scales.to(torch.float32)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_shape = initial_scales.shape
    target_weight = weight.to(torch.float32).reshape(-1, BLOCK_SIZE)
    original_scales = original_scales.to(torch.float32).reshape(-1)
    initial_scales = initial_scales.to(torch.float32).reshape(-1)
    optimized = torch.empty_like(initial_scales)

    for start in range(0, target_weight.shape[0], block_chunk_size):
        end = min(start + block_chunk_size, target_weight.shape[0])
        w_chunk = target_weight[start:end].to(device)
        s_orig = original_scales[start:end].to(device)
        s_init = initial_scales[start:end].to(device)

        init_factor = torch.clamp(s_init / s_orig, min=1.0)
        raw = torch.nn.Parameter(_inverse_softplus(init_factor - 1.0))
        optimizer = torch.optim.Adam([raw], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            factor = F.softplus(raw) + 1.0
            scales = s_orig * factor
            scaled = w_chunk / scales.unsqueeze(-1)
            hard_codes = quantize_to_fp4_codes(scaled.reshape(-1)).view_as(scaled)
            hard_values = fp4_codes_to_values(hard_codes).to(torch.float32)
            ste_values = scaled + (hard_values - scaled).detach()
            dequantized = ste_values * scales.unsqueeze(-1)
            loss = F.mse_loss(dequantized, w_chunk)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            factor = F.softplus(raw) + 1.0
            optimized[start:end] = (s_orig * factor).cpu()

    return optimized.view(output_shape)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.to(torch.float32) - b.to(torch.float32)) ** 2).item())


def _fraction_from_codes(codes: torch.Tensor, predicate: torch.Tensor) -> float:
    return float(predicate.to(torch.float32).mean().item())


def build_layer_metrics(
    *,
    packed: torch.Tensor,
    codes: torch.Tensor,
    dequantized_weight: torch.Tensor,
    baseline_weight: torch.Tensor,
    scales: torch.Tensor,
    reference_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> LayerMetrics:
    entropy_stats = compute_nvfp4_compression_rate(packed.cpu().numpy().astype(np.uint8))
    effective_bits = 4.0 + 8.0 / BLOCK_SIZE + 32.0 / float(baseline_weight.numel())
    scale_ratio = scales / torch.clamp_min(reference_scales, 1e-12)
    zero_mask = (codes == 0) | (codes == 8)
    small_code_mask = zero_mask | (codes == 1) | (codes == 2) | (codes == 9) | (codes == 10)
    return LayerMetrics(
        mean_entropy_bits=entropy_stats["mean_entropy_bits"],
        std_entropy_bits=entropy_stats["std_entropy_bits"],
        compression_rate=entropy_stats["compression_rate"],
        compressed_size_ratio=entropy_stats["compressed_size_ratio"],
        effective_bits_per_weight=effective_bits,
        mse_to_baseline=_mse(dequantized_weight, baseline_weight),
        global_scale=float(global_scale.item()),
        scale_ratio_mean=float(scale_ratio.mean().item()),
        scale_ratio_std=float(scale_ratio.std().item()),
        zero_fraction=_fraction_from_codes(codes, zero_mask),
        small_code_fraction=_fraction_from_codes(codes, small_code_mask),
        n_chunks=entropy_stats["n_chunks"],
    )


def _weight_key_to_prefix(weight_packed_key: str) -> str:
    if not weight_packed_key.endswith(".weight_packed"):
        raise ValueError(f"unexpected packed key: {weight_packed_key}")
    return weight_packed_key[: -len(".weight_packed")]


def _match_any(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def list_nvfp4_weight_prefixes(
    checkpoint_path: str | Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    prefixes = []
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".weight_packed"):
                continue
            prefix = _weight_key_to_prefix(key)
            if not _match_any(prefix, include_patterns):
                continue
            if exclude_patterns and _match_any(prefix, exclude_patterns):
                continue
            prefixes.append(prefix)
    return sorted(prefixes)


def run_scale_inflation_experiment(
    *,
    checkpoint_path: str | Path,
    prefix: str,
    alpha: float = 2.0,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    optimize: bool = True,
    block_chunk_size: int = 131072,
) -> dict[str, Any]:
    """Run scale inflation on one NVFP4 checkpoint tensor."""
    packed_key = prefix + ".weight_packed"
    scale_key = prefix + ".weight_scale"
    global_scale_key = prefix + ".weight_global_scale"

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(packed_key)
        weight_scale = handle.get_tensor(scale_key)
        weight_global_scale = handle.get_tensor(global_scale_key)

    baseline_weight, baseline_actual_scales = dequantize_nvfp4_from_checkpoint(
        packed, weight_scale, weight_global_scale
    )
    baseline_codes = unpack_nvfp4_codes(packed)
    baseline_metrics = build_layer_metrics(
        packed=packed,
        codes=baseline_codes,
        dequantized_weight=baseline_weight,
        baseline_weight=baseline_weight,
        scales=baseline_actual_scales,
        reference_scales=baseline_actual_scales,
        global_scale=weight_global_scale,
    )

    desired_alpha_scales = baseline_actual_scales * float(alpha)
    alpha_scale_fp8, alpha_global_scale = encode_actual_scales(desired_alpha_scales)
    alpha_actual_scales = alpha_scale_fp8.to(torch.float32) / alpha_global_scale
    alpha_packed, alpha_weight, alpha_codes = quantize_to_nvfp4(baseline_weight, alpha_actual_scales)
    alpha_metrics = build_layer_metrics(
        packed=alpha_packed,
        codes=alpha_codes,
        dequantized_weight=alpha_weight,
        baseline_weight=baseline_weight,
        scales=alpha_actual_scales,
        reference_scales=baseline_actual_scales,
        global_scale=alpha_global_scale,
    )

    result: dict[str, Any] = {
        "layer": prefix,
        "shape": list(baseline_weight.shape),
        "num_weights": int(baseline_weight.numel()),
        "baseline": asdict(baseline_metrics),
        "alpha_only": asdict(alpha_metrics),
        "optimized": None,
    }

    if optimize:
        optimized_scales_continuous = optimize_scales_ste(
            baseline_weight,
            baseline_actual_scales,
            alpha_actual_scales,
            steps=steps,
            lr=lr,
            device=device,
            block_chunk_size=block_chunk_size,
        )
        optimized_scale_fp8, optimized_global_scale = encode_actual_scales(optimized_scales_continuous)
        optimized_actual_scales = optimized_scale_fp8.to(torch.float32) / optimized_global_scale
        optimized_packed, optimized_weight, optimized_codes = quantize_to_nvfp4(
            baseline_weight, optimized_actual_scales
        )
        optimized_metrics = build_layer_metrics(
            packed=optimized_packed,
            codes=optimized_codes,
            dequantized_weight=optimized_weight,
            baseline_weight=baseline_weight,
            scales=optimized_actual_scales,
            reference_scales=baseline_actual_scales,
            global_scale=optimized_global_scale,
        )
        result["optimized"] = asdict(optimized_metrics)

    return result


def _weighted_average(results: list[dict[str, Any]], variant: str, field: str) -> float:
    weighted_sum = 0.0
    total = 0
    for result in results:
        metrics = result.get(variant)
        if metrics is None:
            continue
        weight = result["num_weights"]
        weighted_sum += metrics[field] * weight
        total += weight
    return weighted_sum / total if total else 0.0


def build_aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_layers": len(results),
        "num_optimized_layers": sum(1 for result in results if result["optimized"] is not None),
        "baseline_weighted_compression_rate": _weighted_average(
            results, "baseline", "compression_rate"
        ),
        "alpha_only_weighted_compression_rate": _weighted_average(
            results, "alpha_only", "compression_rate"
        ),
        "baseline_weighted_mse": _weighted_average(results, "baseline", "mse_to_baseline"),
        "alpha_only_weighted_mse": _weighted_average(results, "alpha_only", "mse_to_baseline"),
    }
    if summary["num_optimized_layers"] > 0:
        summary["optimized_weighted_compression_rate"] = _weighted_average(
            results, "optimized", "compression_rate"
        )
        summary["optimized_weighted_mse"] = _weighted_average(
            results, "optimized", "mse_to_baseline"
        )
    return summary


def _print_layer_summary(result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    alpha_only = result["alpha_only"]
    text = (
        f"{result['layer'][-64:]:<64} "
        f"base={baseline['compression_rate'] * 100:5.1f}% "
        f"alpha={alpha_only['compression_rate'] * 100:5.1f}% "
        f"alpha_mse={alpha_only['mse_to_baseline']:.3e}"
    )
    if result["optimized"] is not None:
        optimized = result["optimized"]
        text += (
            f" opt={optimized['compression_rate'] * 100:5.1f}% "
            f"opt_mse={optimized['mse_to_baseline']:.3e}"
        )
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model.safetensors")
    parser.add_argument("--alpha", type=float, default=2.0, help="Global scale inflation factor")
    parser.add_argument("--steps", type=int, default=200, help="Adam steps for scale refinement")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--device", default="cuda", help="Optimization device")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern for layers to include. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern for layers to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Limit the number of processed layers. 0 means all matched layers.",
    )
    parser.add_argument(
        "--optimize-max-layers",
        type=int,
        default=0,
        help="Only run Adam optimization on the first N matched layers. 0 means none.",
    )
    parser.add_argument(
        "--block-chunk-size",
        type=int,
        default=131072,
        help="Number of 16-weight blocks optimized together in one chunk.",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    prefixes = list_nvfp4_weight_prefixes(
        checkpoint_path,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )
    if args.max_layers > 0:
        prefixes = prefixes[: args.max_layers]

    results = []
    for index, prefix in enumerate(prefixes):
        optimize = args.optimize_max_layers > 0 and index < args.optimize_max_layers
        result = run_scale_inflation_experiment(
            checkpoint_path=checkpoint_path,
            prefix=prefix,
            alpha=args.alpha,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            optimize=optimize,
            block_chunk_size=args.block_chunk_size,
        )
        results.append(result)
        _print_layer_summary(result)

    summary = build_aggregate_summary(results)
    print("\nAggregate:")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        payload = {
            "checkpoint": str(checkpoint_path),
            "alpha": args.alpha,
            "steps": args.steps,
            "lr": args.lr,
            "device": args.device,
            "include": args.include,
            "exclude": args.exclude,
            "summary": summary,
            "layers": results,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON report to {output_path}")


if __name__ == "__main__":
    main()
