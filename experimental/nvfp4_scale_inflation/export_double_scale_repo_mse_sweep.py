# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a compressed-tensors NVFP4 checkpoint using double-scale repo-style MSE sweep."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import (
    DEFAULT_PRESET,
    DoubleScaleRepoMSESweepConfig,
    ShardedTensorLoader,
    absmax_actual_scales,
    config_with_overrides,
    legalize_scales_with_fixed_global,
    _metrics_from_result,
    _quantize_from_scales,
    preset_config,
    quantize_weight_double_scale_repo_mse_sweep,
)
from experimental.nvfp4_scale_inflation.scale_inflation import list_nvfp4_weight_prefixes


def _parse_size_bytes(size: int | str) -> int:
    if isinstance(size, int):
        return size
    text = str(size).strip().upper()
    units = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "B": 1,
    }
    for suffix, factor in units.items():
        if text.endswith(suffix):
            value = float(text[: -len(suffix)])
            return max(1, int(value * factor))
    return int(text)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _copy_non_weight_files(source_dir: Path, output_dir: Path) -> None:
    ignore = shutil.ignore_patterns("model.safetensors", "model.safetensors.index.json", ".git")
    shutil.copytree(source_dir, output_dir, ignore=ignore)


def _save_sharded_state_dict(
    *,
    output_dir: Path,
    entries: list[tuple[str, torch.Tensor]],
    metadata: dict[str, str] | None,
    max_shard_size: int | str,
) -> list[str]:
    max_bytes = _parse_size_bytes(max_shard_size)
    shard_names: list[str] = []
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_size = 0
    shard_index = 0
    total_size = 0
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal shard_tensors, shard_size, shard_index
        if not shard_tensors:
            return
        shard_index += 1
        shard_name = f"model-{shard_index:05d}.safetensors"
        save_file(shard_tensors, str(output_dir / shard_name), metadata=metadata)
        shard_names.append(shard_name)
        for key in shard_tensors:
            weight_map[key] = shard_name
        shard_tensors = {}
        shard_size = 0

    for key, tensor in entries:
        tensor_size = _tensor_nbytes(tensor)
        total_size += tensor_size
        if shard_tensors and shard_size + tensor_size > max_bytes:
            flush()
        shard_tensors[key] = tensor
        shard_size += tensor_size
    flush()

    if len(shard_names) == 1:
        only_path = output_dir / shard_names[0]
        final_path = output_dir / "model.safetensors"
        only_path.rename(final_path)
        return ["model.safetensors"]

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return shard_names


def export_double_scale_repo_mse_sweep_checkpoint(
    *,
    full_precision_model_dir: str | Path,
    template_nvfp4_dir: str | Path,
    output_dir: str | Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_layers: int = 0,
    row_limit: int = 0,
    config: DoubleScaleRepoMSESweepConfig | None = None,
    max_shard_size: int | str = "1GB",
    device: str = "cpu",
) -> dict[str, Any]:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    config = config or DoubleScaleRepoMSESweepConfig()

    full_precision_model_dir = Path(full_precision_model_dir)
    template_nvfp4_dir = Path(template_nvfp4_dir)
    output_dir = Path(output_dir)
    template_safetensors = template_nvfp4_dir / "model.safetensors"

    if output_dir.exists():
        raise FileExistsError(f"output dir already exists: {output_dir}")

    _copy_non_weight_files(template_nvfp4_dir, output_dir)

    prefixes = list_nvfp4_weight_prefixes(
        template_safetensors,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    if max_layers > 0:
        prefixes = prefixes[:max_layers]

    reports = []

    with safe_open(str(template_safetensors), framework="pt", device="cpu") as template_handle:
        metadata = template_handle.metadata()
        modified_prefixes = set(prefixes)
        pending_modified: dict[str, dict[str, torch.Tensor]] = {}
        ordered_entries: list[tuple[str, torch.Tensor]] = []

        with ShardedTensorLoader(full_precision_model_dir) as fp_loader:
            for key in template_handle.keys():
                prefix = None
                if key.endswith(".weight_packed"):
                    prefix = key[: -len(".weight_packed")]
                elif key.endswith(".weight_scale"):
                    prefix = key[: -len(".weight_scale")]
                elif key.endswith(".weight_global_scale"):
                    prefix = key[: -len(".weight_global_scale")]

                if prefix is not None and prefix in modified_prefixes:
                    if prefix not in pending_modified:
                        full_weight = fp_loader.get_tensor(prefix + ".weight").to(
                            device=device, dtype=torch.float32
                        )
                        if row_limit > 0:
                            raise ValueError(
                                "row_limit export is not supported; export requires full tensors."
                            )
                        absmax_scales = absmax_actual_scales(full_weight)
                        absmax_result = _quantize_from_scales(full_weight, absmax_scales)
                        _, _, double_actual_scales = legalize_scales_with_fixed_global(
                            absmax_scales * config.alpha,
                            global_max_scale=(absmax_scales.max() * config.alpha),
                        )
                        double_result = _quantize_from_scales(full_weight, double_actual_scales)
                        result = quantize_weight_double_scale_repo_mse_sweep(full_weight, config=config)
                        pending_modified[prefix] = {
                            prefix + ".weight_packed": result.packed.cpu(),
                            prefix + ".weight_scale": result.encoded_scale.cpu(),
                            prefix + ".weight_global_scale": result.encoded_global_scale.reshape(1).cpu(),
                        }
                        absmax_metrics = _metrics_from_result(full_weight, absmax_result)
                        double_metrics = _metrics_from_result(full_weight, double_result)
                        final_metrics = _metrics_from_result(full_weight, result)
                        reports.append(
                            {
                                "layer": prefix,
                                "shape": list(full_weight.shape),
                                "num_weights": int(full_weight.numel()),
                                "alpha": config.alpha,
                                "absmax_before": absmax_metrics,
                                "double_scale_before": double_metrics,
                                "final_after": final_metrics,
                            }
                        )
                        print(
                            f"exported {prefix} "
                            f"abs_mse={absmax_metrics['mse']:.6e} "
                            f"abs_entropy={absmax_metrics['mean_entropy_bits']:.4f}bit "
                            f"double_mse={double_metrics['mse']:.6e} "
                            f"double_entropy={double_metrics['mean_entropy_bits']:.4f}bit "
                            f"final_mse={final_metrics['mse']:.6e} "
                            f"final_entropy={final_metrics['mean_entropy_bits']:.4f}bit "
                            f"final_comp={final_metrics['compression_rate'] * 100:.2f}%"
                        )
                        if device.startswith("cuda"):
                            del full_weight, absmax_result, double_result, result
                            torch.cuda.empty_cache()

                    tensor = pending_modified[prefix].pop(key)
                    ordered_entries.append((key, tensor))
                    if not pending_modified[prefix]:
                        del pending_modified[prefix]
                else:
                    ordered_entries.append((key, template_handle.get_tensor(key)))

    shard_names = _save_sharded_state_dict(
        output_dir=output_dir,
        entries=ordered_entries,
        metadata=metadata,
        max_shard_size=max_shard_size,
    )

    summary = {
        "full_precision_model_dir": str(full_precision_model_dir),
        "template_nvfp4_dir": str(template_nvfp4_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "num_modified_layers": len(prefixes),
        "aggregate": {
            "absmax_before_weighted_mse": _weighted_average(reports, "absmax_before", "mse"),
            "absmax_before_weighted_entropy_bits": _weighted_average(
                reports, "absmax_before", "mean_entropy_bits"
            ),
            "absmax_before_weighted_compression_rate": _weighted_average(
                reports, "absmax_before", "compression_rate"
            ),
            "double_scale_before_weighted_mse": _weighted_average(reports, "double_scale_before", "mse"),
            "double_scale_before_weighted_entropy_bits": _weighted_average(
                reports, "double_scale_before", "mean_entropy_bits"
            ),
            "double_scale_before_weighted_compression_rate": _weighted_average(
                reports, "double_scale_before", "compression_rate"
            ),
            "final_after_weighted_mse": _weighted_average(reports, "final_after", "mse"),
            "final_after_weighted_entropy_bits": _weighted_average(
                reports, "final_after", "mean_entropy_bits"
            ),
            "final_after_weighted_compression_rate": _weighted_average(
                reports, "final_after", "compression_rate"
            ),
        },
        "layers": reports,
        "shards": shard_names,
        "max_shard_size": str(max_shard_size),
    }
    (output_dir / "double_scale_repo_mse_sweep_export.json").write_text(json.dumps(summary, indent=2))
    print(
        "aggregate "
        f"abs_mse={summary['aggregate']['absmax_before_weighted_mse']:.6e} "
        f"abs_entropy={summary['aggregate']['absmax_before_weighted_entropy_bits']:.4f}bit "
        f"double_mse={summary['aggregate']['double_scale_before_weighted_mse']:.6e} "
        f"double_entropy={summary['aggregate']['double_scale_before_weighted_entropy_bits']:.4f}bit "
        f"final_mse={summary['aggregate']['final_after_weighted_mse']:.6e} "
        f"final_entropy={summary['aggregate']['final_after_weighted_entropy_bits']:.4f}bit "
        f"final_comp={summary['aggregate']['final_after_weighted_compression_rate'] * 100:.2f}%"
    )
    return summary


def _weighted_average(reports: list[dict[str, Any]], key: str, field: str) -> float | None:
    total_weight = 0
    total = 0.0
    for report in reports:
        item = report.get(key)
        if item is None:
            continue
        num_weights = int(report["num_weights"])
        total_weight += num_weights
        total += num_weights * float(item[field])
    if total_weight == 0:
        return None
    return total / total_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-precision-model-dir", required=True)
    parser.add_argument("--template-nvfp4-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-layers", type=int, default=0)
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
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_double_scale_repo_mse_sweep_checkpoint(
        full_precision_model_dir=args.full_precision_model_dir,
        template_nvfp4_dir=args.template_nvfp4_dir,
        output_dir=args.output_dir,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_layers=args.max_layers,
        config=config_with_overrides(
            preset_config(args.preset),
            alpha=args.alpha,
            min_scale_ratio=args.min_scale_ratio,
            soft_entropy_budget_ratio=args.soft_entropy_budget_ratio,
            soft_temperature=args.soft_temperature,
            candidate_chunk_size=args.candidate_chunk_size,
            block_chunk_size=args.block_chunk_size,
        ),
        max_shard_size=args.max_shard_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
