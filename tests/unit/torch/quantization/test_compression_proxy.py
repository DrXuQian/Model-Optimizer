# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for experimental compression-aware QAT proxy helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer

EXAMPLES_LLM_QAT = Path(__file__).resolve().parents[4] / "examples" / "llm_qat"
if str(EXAMPLES_LLM_QAT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_LLM_QAT))

from compression_proxy import (  # noqa: E402
    CompressionProxyConfig,
    collect_trainable_nvfp4_weight_refs,
    compute_compression_proxy,
)


class TinyNVFP4Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, per_block_amax: torch.Tensor, global_amax: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        quant_cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
            enable=True,
        )
        quantizer = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=per_block_amax.clone())
        self.weight_quantizer = NVFP4StaticQuantizer.from_tensor_quantizer(
            quantizer, global_amax=torch.tensor(global_amax, dtype=torch.float32)
        )


def test_collect_trainable_nvfp4_weight_refs_finds_static_weight_quantizer():
    weight = torch.zeros(2, 16, dtype=torch.float32)
    model = TinyNVFP4Linear(weight, per_block_amax=torch.full((2, 1), 6.0), global_amax=6.0)

    refs = collect_trainable_nvfp4_weight_refs(model)

    assert len(refs) == 1
    assert refs[0].name == "weight"
    assert refs[0].weight.requires_grad


def test_soft_code_proxy_prefers_concentrated_code_usage():
    concentrated = TinyNVFP4Linear(
        weight=torch.zeros(2, 16, dtype=torch.float32),
        per_block_amax=torch.full((2, 1), 6.0),
        global_amax=6.0,
    )
    spread = TinyNVFP4Linear(
        weight=torch.tensor(
            [
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            ],
            dtype=torch.float32,
        ),
        per_block_amax=torch.full((2, 1), 6.0),
        global_amax=6.0,
    )
    config = CompressionProxyConfig(
        proxy_type="soft_code",
        temperature=0.1,
        sample_layers=0,
        sample_blocks_per_layer=0,
        seed=1234,
    )

    concentrated_stats = compute_compression_proxy(
        collect_trainable_nvfp4_weight_refs(concentrated), config, step=0
    )
    spread_stats = compute_compression_proxy(
        collect_trainable_nvfp4_weight_refs(spread), config, step=0
    )

    assert concentrated_stats.entropy_bits.item() < spread_stats.entropy_bits.item()
    assert concentrated_stats.num_values_sampled == spread_stats.num_values_sampled == 32
