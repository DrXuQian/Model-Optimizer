# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a modified NVFP4 checkpoint using full-precision source weights as the target."""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .scale_inflation import (
    build_aggregate_summary,
    build_layer_metrics,
    dequantize_nvfp4_from_checkpoint,
    encode_actual_scales,
    list_nvfp4_weight_prefixes,
    optimize_scales_ste,
    quantize_to_nvfp4,
    unpack_nvfp4_codes,
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
        shard_files = sorted(set(self._weight_map.values()))
        for shard_file in shard_files:
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


def _match_any(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def export_from_full_precision(
    *,
    full_precision_model_dir: str | Path,
    template_nvfp4_dir: str | Path,
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
    full_precision_model_dir = Path(full_precision_model_dir)
    template_nvfp4_dir = Path(template_nvfp4_dir)
    output_dir = Path(output_dir)
    template_safetensors = template_nvfp4_dir / "model.safetensors"
    output_safetensors = output_dir / "model.safetensors"

    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    if output_dir.exists():
        raise FileExistsError(f"output dir already exists: {output_dir}")

    shutil.copytree(template_nvfp4_dir, output_dir)

    prefixes = list_nvfp4_weight_prefixes(
        template_safetensors,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    modified_tensors: dict[str, torch.Tensor] = {}
    layer_reports = []

    with safe_open(str(template_safetensors), framework="pt", device="cpu") as template_handle:
        metadata = template_handle.metadata()
        with ShardedTensorLoader(full_precision_model_dir) as fp_loader:
            for index, prefix in enumerate(prefixes):
                packed_key = prefix + ".weight_packed"
                scale_key = prefix + ".weight_scale"
                global_scale_key = prefix + ".weight_global_scale"
                weight_key = prefix + ".weight"

                template_packed = template_handle.get_tensor(packed_key)
                template_scale = template_handle.get_tensor(scale_key)
                template_global_scale = template_handle.get_tensor(global_scale_key)
                full_weight = fp_loader.get_tensor(weight_key).to(torch.float32)

                baseline_weight, baseline_actual_scales = dequantize_nvfp4_from_checkpoint(
                    template_packed,
                    template_scale,
                    template_global_scale,
                )
                baseline_codes = unpack_nvfp4_codes(template_packed)
                baseline_metrics = build_layer_metrics(
                    packed=template_packed,
                    codes=baseline_codes,
                    dequantized_weight=baseline_weight,
                    baseline_weight=full_weight,
                    scales=baseline_actual_scales,
                    reference_scales=baseline_actual_scales,
                    global_scale=template_global_scale,
                )

                desired_alpha_scales = baseline_actual_scales * float(alpha)
                alpha_scale_fp8, alpha_global_scale = encode_actual_scales(desired_alpha_scales)
                alpha_actual_scales = alpha_scale_fp8.to(torch.float32) / alpha_global_scale
                alpha_packed, alpha_weight, alpha_codes = quantize_to_nvfp4(
                    full_weight, alpha_actual_scales
                )
                alpha_metrics = build_layer_metrics(
                    packed=alpha_packed,
                    codes=alpha_codes,
                    dequantized_weight=alpha_weight,
                    baseline_weight=full_weight,
                    scales=alpha_actual_scales,
                    reference_scales=baseline_actual_scales,
                    global_scale=alpha_global_scale,
                )

                report = {
                    "layer": prefix,
                    "shape": list(full_weight.shape),
                    "num_weights": int(full_weight.numel()),
                    "baseline": asdict(baseline_metrics),
                    "alpha_only": asdict(alpha_metrics),
                    "optimized": None,
                }

                export_packed = alpha_packed
                export_scale = alpha_scale_fp8
                export_global_scale = alpha_global_scale
                export_variant = "alpha_only"

                if optimize_max_layers > 0 and index < optimize_max_layers:
                    optimized_scales_continuous = optimize_scales_ste(
                        full_weight,
                        baseline_actual_scales,
                        desired_alpha_scales,
                        steps=steps,
                        lr=lr,
                        device=device,
                        block_chunk_size=block_chunk_size,
                    )
                    optimized_scale_fp8, optimized_global_scale = encode_actual_scales(
                        optimized_scales_continuous
                    )
                    optimized_actual_scales = (
                        optimized_scale_fp8.to(torch.float32) / optimized_global_scale
                    )
                    optimized_packed, optimized_weight, optimized_codes = quantize_to_nvfp4(
                        full_weight, optimized_actual_scales
                    )
                    optimized_metrics = build_layer_metrics(
                        packed=optimized_packed,
                        codes=optimized_codes,
                        dequantized_weight=optimized_weight,
                        baseline_weight=full_weight,
                        scales=optimized_actual_scales,
                        reference_scales=baseline_actual_scales,
                        global_scale=optimized_global_scale,
                    )
                    report["optimized"] = asdict(optimized_metrics)
                    export_packed = optimized_packed
                    export_scale = optimized_scale_fp8
                    export_global_scale = optimized_global_scale
                    export_variant = "optimized"

                modified_tensors[packed_key] = export_packed.cpu()
                modified_tensors[scale_key] = export_scale.cpu()
                modified_tensors[global_scale_key] = export_global_scale.reshape_as(
                    template_global_scale
                ).cpu()
                layer_reports.append(report)

                print(
                    f"{prefix[-64:]:<64} "
                    f"base_mse={report['baseline']['mse_to_baseline']:.3e} "
                    f"new_mse={report[export_variant]['mse_to_baseline']:.3e} "
                    f"base_comp={report['baseline']['compression_rate'] * 100:5.1f}% "
                    f"new_comp={report[export_variant]['compression_rate'] * 100:5.1f}%"
                )

            state_dict = {}
            for key in template_handle.keys():
                state_dict[key] = modified_tensors.get(key, template_handle.get_tensor(key))

    save_file(state_dict, str(output_safetensors), metadata=metadata)

    summary = {
        "full_precision_model_dir": str(full_precision_model_dir),
        "template_nvfp4_dir": str(template_nvfp4_dir),
        "output_dir": str(output_dir),
        "alpha": alpha,
        "num_modified_layers": len(prefixes),
        "num_optimized_layers": min(optimize_max_layers, len(prefixes)),
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "aggregate": build_aggregate_summary(layer_reports),
        "layers": layer_reports,
    }
    (output_dir / "nvfp4_scale_inflation_from_full_precision_export.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument("--template-nvfp4-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
    export_from_full_precision(
        full_precision_model_dir=args.full_precision_model_dir,
        template_nvfp4_dir=args.template_nvfp4_dir,
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
