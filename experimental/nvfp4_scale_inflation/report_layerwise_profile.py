# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the resolved per-layer configs for a layerwise NVFP4 compression profile."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import (  # noqa: E402
    config_with_overrides,
    preset_config,
)
from experimental.nvfp4_scale_inflation.layerwise_profile import (  # noqa: E402
    infer_num_transformer_layers,
    resolve_layerwise_config,
)
from experimental.nvfp4_scale_inflation.scale_inflation import list_nvfp4_weight_prefixes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-nvfp4-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--preset",
        choices=("precision", "compress10", "mse"),
        default="compress10",
    )
    parser.add_argument(
        "--layerwise-profile",
        choices=("uniform", "tmo_attention_guarded", "llm_sensitivity_v1"),
        default="llm_sensitivity_v1",
    )
    parser.add_argument("--layerwise-rules-json", default=None)
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
    prefixes = list_nvfp4_weight_prefixes(template_path)
    base_config = config_with_overrides(
        preset_config(args.preset),
        alpha=args.alpha,
        min_scale_ratio=args.min_scale_ratio,
        soft_entropy_budget_ratio=args.soft_entropy_budget_ratio,
        soft_temperature=args.soft_temperature,
        candidate_chunk_size=args.candidate_chunk_size,
        block_chunk_size=args.block_chunk_size,
    )
    num_layers = infer_num_transformer_layers(prefixes)

    rows = []
    rule_counter: Counter[str] = Counter()
    for prefix in prefixes:
        resolved, metadata = resolve_layerwise_config(
            base_config=base_config,
            layer_name=prefix,
            num_transformer_layers=num_layers,
            profile=args.layerwise_profile,
            custom_rules_path=args.layerwise_rules_json,
        )
        applied_names = [rule["name"] for rule in metadata["applied_rules"]]
        rule_counter.update(applied_names)
        rows.append(
            {
                "layer": prefix,
                "resolved_config": asdict(resolved),
                "applied_rules": applied_names,
            }
        )

    payload = {
        "base_config": asdict(base_config),
        "layerwise_profile": args.layerwise_profile,
        "layerwise_rules_json": args.layerwise_rules_json,
        "num_transformer_layers": num_layers,
        "num_layers": len(rows),
        "rule_counts": dict(rule_counter),
        "layers": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output_path}")
    print(f"profile={args.layerwise_profile} num_layers={len(rows)} transformer_layers={num_layers}")
    for name, count in sorted(rule_counter.items()):
        print(f"rule {name}: {count}")


if __name__ == "__main__":
    main()
