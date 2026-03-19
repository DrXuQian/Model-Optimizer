# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a compressed-tensors NVFP4 checkpoint using layer-level global-budget search."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import (
    DEFAULT_PRESET,
    DoubleScaleRepoMSESweepConfig,
    ShardedTensorLoader,
    config_with_overrides,
    preset_config,
)
from experimental.nvfp4_scale_inflation.layerwise_profile import (
    infer_num_transformer_layers,
    resolve_layerwise_config,
)
from experimental.nvfp4_scale_inflation.global_budget_layer_search import (
    quantize_weight_global_budget_repo_mse_sweep,
)
from experimental.nvfp4_scale_inflation.scale_inflation import list_nvfp4_weight_prefixes


def _parse_size_bytes(size: int | str) -> int:
    if isinstance(size, int):
        return size
    text = str(size).strip().upper()
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "B": 1}
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

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return shard_names


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


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt_float(value: float | None, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def export_global_budget_repo_mse_sweep_checkpoint(
    *,
    full_precision_model_dir: str | Path,
    template_nvfp4_dir: str | Path,
    output_dir: str | Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_layers: int = 0,
    config: DoubleScaleRepoMSESweepConfig | None = None,
    max_shard_size: int | str = "1GB",
    device: str = "cpu",
    layerwise_profile: str = "uniform",
    layerwise_rules_json: str | Path | None = None,
    entropy_proxy: str = "soft",
) -> dict[str, Any]:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    config = config or preset_config(DEFAULT_PRESET)

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
    num_transformer_layers = infer_num_transformer_layers(prefixes)

    reports: list[dict[str, Any]] = []
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
                        full_weight = fp_loader.get_tensor(prefix + ".weight").to(device=device, dtype=torch.float32)
                        layer_config, layerwise_metadata = resolve_layerwise_config(
                            base_config=config,
                            layer_name=prefix,
                            num_transformer_layers=num_transformer_layers,
                            profile=layerwise_profile,
                            custom_rules_path=layerwise_rules_json,
                        )
                        quantized = quantize_weight_global_budget_repo_mse_sweep(
                            full_weight,
                            config=layer_config,
                            entropy_proxy=entropy_proxy,
                        )
                        global_result = quantized["global_result"]
                        report = {
                            "layer": prefix,
                            "shape": list(full_weight.shape),
                            "num_weights": int(full_weight.numel()),
                            "config": asdict(layer_config),
                            "base_config": asdict(config),
                            "layerwise_profile": layerwise_metadata,
                            "absmax_before": quantized["report"]["absmax"],
                            "double_scale_before": quantized["report"]["double_scale"],
                            "local_budget": quantized["report"]["local_budget"],
                            "final_after": quantized["report"]["global_budget"]["hard_metrics"],
                            "global_budget": {
                                key: value
                                for key, value in quantized["report"]["global_budget"].items()
                                if key != "hard_metrics"
                            },
                        }
                        reports.append(report)
                        pending_modified[prefix] = {
                            prefix + ".weight_packed": global_result.packed.cpu(),
                            prefix + ".weight_scale": global_result.encoded_scale.cpu(),
                            prefix + ".weight_global_scale": global_result.encoded_global_scale.reshape(1).cpu(),
                        }
                        print(
                            f"exported {prefix} "
                            f"profile={layerwise_profile} "
                            f"abs(comp={report['absmax_before']['compression_rate'] * 100:.2f}% "
                            f"entropy={report['absmax_before']['mean_entropy_bits']:.4f}bit "
                            f"mse={report['absmax_before']['mse']:.6e}) "
                            f"double(comp={report['double_scale_before']['compression_rate'] * 100:.2f}% "
                            f"entropy={report['double_scale_before']['mean_entropy_bits']:.4f}bit "
                            f"mse={report['double_scale_before']['mse']:.6e}) "
                            f"local(comp={report['local_budget']['compression_rate'] * 100:.2f}% "
                            f"entropy={report['local_budget']['mean_entropy_bits']:.4f}bit "
                            f"mse={report['local_budget']['mse']:.6e}) "
                            f"final(comp={report['final_after']['compression_rate'] * 100:.2f}% "
                            f"entropy={report['final_after']['mean_entropy_bits']:.4f}bit "
                            f"mse={report['final_after']['mse']:.6e})",
                            flush=True,
                        )
                        if device.startswith("cuda"):
                            del full_weight, quantized, global_result
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
        "layerwise_profile": layerwise_profile,
        "layerwise_rules_json": None if layerwise_rules_json is None else str(layerwise_rules_json),
        "num_transformer_layers": num_transformer_layers,
        "num_modified_layers": len(prefixes),
        "aggregate": {
            "absmax_before_weighted_mse": _weighted_average(reports, "absmax_before", "mse"),
            "absmax_before_weighted_mean_entropy_bits": _weighted_average(reports, "absmax_before", "mean_entropy_bits"),
            "absmax_before_weighted_compression_rate": _weighted_average(reports, "absmax_before", "compression_rate"),
            "double_scale_before_weighted_mse": _weighted_average(reports, "double_scale_before", "mse"),
            "double_scale_before_weighted_mean_entropy_bits": _weighted_average(
                reports, "double_scale_before", "mean_entropy_bits"
            ),
            "double_scale_before_weighted_compression_rate": _weighted_average(
                reports, "double_scale_before", "compression_rate"
            ),
            "local_budget_weighted_mse": _weighted_average(reports, "local_budget", "mse"),
            "local_budget_weighted_mean_entropy_bits": _weighted_average(
                reports, "local_budget", "mean_entropy_bits"
            ),
            "local_budget_weighted_compression_rate": _weighted_average(reports, "local_budget", "compression_rate"),
            "final_after_weighted_mse": _weighted_average(reports, "final_after", "mse"),
            "final_after_weighted_mean_entropy_bits": _weighted_average(reports, "final_after", "mean_entropy_bits"),
            "final_after_weighted_compression_rate": _weighted_average(reports, "final_after", "compression_rate"),
        },
        "layers": reports,
        "shards": shard_names,
        "max_shard_size": str(max_shard_size),
    }
    (output_dir / "global_budget_repo_mse_sweep_export.json").write_text(json.dumps(summary, indent=2))
    print(
        "aggregate "
        f"abs(comp={_fmt_pct(summary['aggregate']['absmax_before_weighted_compression_rate'])} "
        f"entropy={_fmt_float(summary['aggregate']['absmax_before_weighted_mean_entropy_bits'])}bit "
        f"mse={_fmt_sci(summary['aggregate']['absmax_before_weighted_mse'])}) "
        f"double(comp={_fmt_pct(summary['aggregate']['double_scale_before_weighted_compression_rate'])} "
        f"entropy={_fmt_float(summary['aggregate']['double_scale_before_weighted_mean_entropy_bits'])}bit "
        f"mse={_fmt_sci(summary['aggregate']['double_scale_before_weighted_mse'])}) "
        f"local(comp={_fmt_pct(summary['aggregate']['local_budget_weighted_compression_rate'])} "
        f"entropy={_fmt_float(summary['aggregate']['local_budget_weighted_mean_entropy_bits'])}bit "
        f"mse={_fmt_sci(summary['aggregate']['local_budget_weighted_mse'])}) "
        f"final(comp={_fmt_pct(summary['aggregate']['final_after_weighted_compression_rate'])} "
        f"entropy={_fmt_float(summary['aggregate']['final_after_weighted_mean_entropy_bits'])}bit "
        f"mse={_fmt_sci(summary['aggregate']['final_after_weighted_mse'])})",
        flush=True,
    )
    print(f"wrote report {output_dir / 'global_budget_repo_mse_sweep_export.json'}", flush=True)
    return summary


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
    parser.add_argument(
        "--layerwise-profile",
        choices=("uniform", "tmo_attention_guarded", "llm_sensitivity_v1"),
        default="uniform",
        help="Optional layer-wise compression schedule applied on top of the base preset.",
    )
    parser.add_argument(
        "--layerwise-rules-json",
        default=None,
        help="Optional JSON file with extra custom layerwise rules.",
    )
    parser.add_argument(
        "--entropy-proxy",
        choices=("soft", "hard"),
        default="soft",
        help="Entropy proxy for search: 'soft' (default) or 'hard' (frequency-based).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_global_budget_repo_mse_sweep_checkpoint(
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
        layerwise_profile=args.layerwise_profile,
        layerwise_rules_json=args.layerwise_rules_json,
        entropy_proxy=args.entropy_proxy,
    )


if __name__ == "__main__":
    main()
