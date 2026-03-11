# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a modified compressed-tensors NVFP4 checkpoint."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .scale_inflation import (
    encode_actual_scales,
    list_nvfp4_weight_prefixes,
    optimize_scales_ste,
    quantize_to_nvfp4,
    dequantize_nvfp4_from_checkpoint,
    run_scale_inflation_experiment,
)


def _match_any(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def export_modified_checkpoint(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 2.0,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    optimize_max_layers: int = 0,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    block_chunk_size: int = 131072,
) -> dict:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_safetensors = source_dir / "model.safetensors"
    output_safetensors = output_dir / "model.safetensors"

    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    if output_dir.exists():
        raise FileExistsError(f"output dir already exists: {output_dir}")

    shutil.copytree(source_dir, output_dir)

    prefixes = list_nvfp4_weight_prefixes(
        source_safetensors,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    modified_tensors: dict[str, torch.Tensor] = {}
    layer_reports = []

    with safe_open(str(source_safetensors), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for index, prefix in enumerate(prefixes):
            packed_key = prefix + ".weight_packed"
            scale_key = prefix + ".weight_scale"
            global_scale_key = prefix + ".weight_global_scale"

            packed = handle.get_tensor(packed_key)
            weight_scale = handle.get_tensor(scale_key)
            weight_global_scale = handle.get_tensor(global_scale_key)

            baseline_weight, baseline_actual_scales = dequantize_nvfp4_from_checkpoint(
                packed,
                weight_scale,
                weight_global_scale,
            )

            actual_scales = baseline_actual_scales * float(alpha)
            if optimize_max_layers > 0 and index < optimize_max_layers:
                actual_scales = optimize_scales_ste(
                    baseline_weight,
                    baseline_actual_scales,
                    actual_scales,
                    steps=steps,
                    lr=lr,
                    device=device,
                    block_chunk_size=block_chunk_size,
                )

            encoded_scale, encoded_global_scale = encode_actual_scales(actual_scales)
            repacked, _, _ = quantize_to_nvfp4(
                baseline_weight, encoded_scale.to(torch.float32) / encoded_global_scale
            )

            modified_tensors[packed_key] = repacked.cpu()
            modified_tensors[scale_key] = encoded_scale.cpu()
            modified_tensors[global_scale_key] = encoded_global_scale.cpu()

            report = run_scale_inflation_experiment(
                checkpoint_path=source_safetensors,
                prefix=prefix,
                alpha=alpha,
                steps=steps,
                lr=lr,
                device=device,
                optimize=optimize_max_layers > 0 and index < optimize_max_layers,
                block_chunk_size=block_chunk_size,
            )
            layer_reports.append(report)
            variant = "optimized" if report["optimized"] is not None else "alpha_only"
            print(
                f"{prefix[-64:]:<64} "
                f"base={report['baseline']['compression_rate'] * 100:5.1f}% "
                f"new={report[variant]['compression_rate'] * 100:5.1f}% "
                f"mse={report[variant]['mse_to_baseline']:.3e}"
            )

        state_dict = {}
        for key in handle.keys():
            if key in modified_tensors:
                state_dict[key] = modified_tensors[key]
            else:
                state_dict[key] = handle.get_tensor(key)

    save_file(state_dict, str(output_safetensors), metadata=metadata)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "alpha": alpha,
        "num_modified_layers": len(prefixes),
        "num_optimized_layers": min(optimize_max_layers, len(prefixes)),
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "layers": layer_reports,
    }
    (output_dir / "nvfp4_scale_inflation_export.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Source HF checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Output HF checkpoint directory")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--optimize-max-layers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-chunk-size", type=int, default=131072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_modified_checkpoint(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        optimize_max_layers=args.optimize_max_layers,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        block_chunk_size=args.block_chunk_size,
    )


if __name__ == "__main__":
    main()
