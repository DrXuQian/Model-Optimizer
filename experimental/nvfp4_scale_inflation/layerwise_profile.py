# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise compression schedules for NVFP4 export."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experimental.nvfp4_scale_inflation.double_scale_repo_mse_sweep import DoubleScaleRepoMSESweepConfig

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class LayerwiseCompressionRule:
    name: str
    patterns: tuple[str, ...] = ("*",)
    min_layer: int | None = None
    max_layer: int | None = None
    edge_layers: int | None = None
    alpha_scale: float = 1.0
    soft_budget_scale: float | None = None
    soft_budget_override: float | None = None
    min_scale_ratio_override: float | None = None
    description: str = ""


def _layer_index(layer_name: str) -> int | None:
    match = _LAYER_RE.search(layer_name)
    return int(match.group(1)) if match else None


def infer_num_transformer_layers(prefixes: list[str]) -> int | None:
    indices = [_layer_index(prefix) for prefix in prefixes]
    indices = [idx for idx in indices if idx is not None]
    if not indices:
        return None
    return max(indices) + 1


def _matches_rule(
    rule: LayerwiseCompressionRule,
    *,
    layer_name: str,
    layer_idx: int | None,
    num_layers: int | None,
) -> bool:
    if not any(fnmatch.fnmatch(layer_name, pattern) for pattern in rule.patterns):
        return False
    if rule.min_layer is not None and (layer_idx is None or layer_idx < rule.min_layer):
        return False
    if rule.max_layer is not None and (layer_idx is None or layer_idx > rule.max_layer):
        return False
    if rule.edge_layers is not None:
        if layer_idx is None or num_layers is None:
            return False
        if not (layer_idx < rule.edge_layers or layer_idx >= num_layers - rule.edge_layers):
            return False
    return True


def _builtin_profile_rules(profile: str) -> list[LayerwiseCompressionRule]:
    name = profile.strip().lower()
    if name == "uniform":
        return []
    if name == "tmo_attention_guarded":
        return [
            LayerwiseCompressionRule(
                name="qkv_conservative",
                patterns=(
                    "*.self_attn.q_proj",
                    "*.self_attn.k_proj",
                    "*.self_attn.v_proj",
                ),
                alpha_scale=0.85,
                soft_budget_override=0.75,
                description="Mimic TMO's NVFP4_MLP_ONLY/OMLP_ONLY preference by keeping QKV more conservative.",
            ),
            LayerwiseCompressionRule(
                name="o_proj_moderate",
                patterns=("*.self_attn.o_proj",),
                alpha_scale=0.95,
                soft_budget_override=0.50,
                description="o_proj is less sensitive than QKV but still more sensitive than MLP.",
            ),
            LayerwiseCompressionRule(
                name="mlp_slightly_aggressive",
                patterns=("*.mlp.*_proj",),
                soft_budget_scale=0.90,
                description="MLP projections typically tolerate slightly more compression.",
            ),
        ]
    if name == "llm_sensitivity_v1":
        return [
            *_builtin_profile_rules("tmo_attention_guarded"),
            LayerwiseCompressionRule(
                name="edge_layer_guard",
                patterns=("model.layers.*.*",),
                edge_layers=4,
                alpha_scale=0.90,
                soft_budget_scale=1.35,
                description="First/last transformer blocks are often more sensitive; back off compression there.",
            ),
            LayerwiseCompressionRule(
                name="middle_mlp_bonus",
                patterns=("*.mlp.*_proj",),
                min_layer=4,
                max_layer=27,
                soft_budget_scale=0.85,
                description="Recover compression in the middle MLP stack where sensitivity is typically lower.",
            ),
        ]
    raise ValueError(f"Unsupported layerwise profile: {profile}")


def load_custom_layerwise_rules(path: str | Path) -> list[LayerwiseCompressionRule]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise TypeError("layerwise rules JSON must be a list of rule objects")
    rules = []
    for item in payload:
        rules.append(
            LayerwiseCompressionRule(
                name=item["name"],
                patterns=tuple(item.get("patterns", ["*"])),
                min_layer=item.get("min_layer"),
                max_layer=item.get("max_layer"),
                edge_layers=item.get("edge_layers"),
                alpha_scale=float(item.get("alpha_scale", 1.0)),
                soft_budget_scale=item.get("soft_budget_scale"),
                soft_budget_override=item.get("soft_budget_override"),
                min_scale_ratio_override=item.get("min_scale_ratio_override"),
                description=item.get("description", ""),
            )
        )
    return rules


def resolve_layerwise_config(
    *,
    base_config: DoubleScaleRepoMSESweepConfig,
    layer_name: str,
    num_transformer_layers: int | None,
    profile: str = "uniform",
    custom_rules_path: str | Path | None = None,
) -> tuple[DoubleScaleRepoMSESweepConfig, dict[str, Any]]:
    layer_idx = _layer_index(layer_name)
    rules = _builtin_profile_rules(profile)
    if custom_rules_path is not None:
        rules.extend(load_custom_layerwise_rules(custom_rules_path))

    alpha = float(base_config.alpha)
    soft_budget = base_config.soft_entropy_budget_ratio
    min_scale_ratio = float(base_config.min_scale_ratio)
    applied_rules: list[dict[str, Any]] = []

    for rule in rules:
        if not _matches_rule(rule, layer_name=layer_name, layer_idx=layer_idx, num_layers=num_transformer_layers):
            continue
        alpha *= float(rule.alpha_scale)
        if rule.soft_budget_override is not None:
            soft_budget = float(rule.soft_budget_override)
        elif rule.soft_budget_scale is not None and soft_budget is not None:
            soft_budget *= float(rule.soft_budget_scale)
        if rule.min_scale_ratio_override is not None:
            min_scale_ratio = float(rule.min_scale_ratio_override)
        applied_rules.append(asdict(rule))

    if soft_budget is not None:
        soft_budget = min(max(float(soft_budget), 0.0), 1.0)

    resolved = DoubleScaleRepoMSESweepConfig(
        alpha=max(alpha, 1e-6),
        min_scale_ratio=max(min_scale_ratio, 1e-6),
        soft_entropy_budget_ratio=soft_budget,
        soft_temperature=base_config.soft_temperature,
        candidate_chunk_size=base_config.candidate_chunk_size,
        block_chunk_size=base_config.block_chunk_size,
    )
    metadata = {
        "profile": profile,
        "custom_rules_path": None if custom_rules_path is None else str(custom_rules_path),
        "layer_index": layer_idx,
        "num_transformer_layers": num_transformer_layers,
        "applied_rules": applied_rules,
        "resolved_config": asdict(resolved),
    }
    return resolved, metadata
