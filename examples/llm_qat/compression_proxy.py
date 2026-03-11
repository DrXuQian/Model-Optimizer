# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compression proxy utilities for experimental compression-aware QAT."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.tensor_quant import scaled_e4m3_impl
from modelopt.torch.quantization.utils import quantizer_attr_names, weight_attr_names


@dataclass
class CompressionProxyConfig:
    """Configuration for experimental compression proxy regularization."""

    proxy_type: str = "soft_code"
    temperature: float = 0.25
    sample_layers: int = 0
    sample_blocks_per_layer: int = 256
    seed: int = 1234
    layer_name_filter: str = ""


@dataclass
class CompressionProxyStats:
    """Aggregated compression proxy statistics."""

    entropy_bits: torch.Tensor
    num_layers_sampled: int
    num_blocks_sampled: int
    num_values_sampled: int


@dataclass
class NVFP4WeightRef:
    """Reference to a trainable NVFP4 weight quantizer."""

    name: str
    module: torch.nn.Module
    weight_name: str
    weight: torch.nn.Parameter
    quantizer: TensorQuantizer


def collect_trainable_nvfp4_weight_refs(model: torch.nn.Module) -> list[NVFP4WeightRef]:
    """Collect trainable static NVFP4 weight quantizers from a quantized model."""

    refs: list[NVFP4WeightRef] = []
    name_to_module = dict(model.named_modules())
    seen_modules = set()

    for module_name, module in name_to_module.items():
        if module in seen_modules:
            continue
        for weight_name in weight_attr_names(module):
            qname = quantizer_attr_names(weight_name).weight_quantizer
            quantizer = getattr(module, qname, None)
            weight = getattr(module, weight_name, None)
            if not isinstance(quantizer, TensorQuantizer):
                continue
            if weight is None or not isinstance(weight, torch.nn.Parameter) or not weight.requires_grad:
                continue
            if not _is_static_nvfp4_weight_quantizer(quantizer):
                continue
            ref_name = f"{module_name}.{weight_name}" if module_name else weight_name
            refs.append(
                NVFP4WeightRef(
                    name=ref_name,
                    module=module,
                    weight_name=weight_name,
                    weight=weight,
                    quantizer=quantizer,
                )
            )
        seen_modules.add(module)

    return refs


def compute_compression_proxy(
    refs: list[NVFP4WeightRef],
    config: CompressionProxyConfig,
    step: int = 0,
) -> CompressionProxyStats:
    """Compute an experimental compression proxy on sampled NVFP4 weights."""

    if not refs:
        raise ValueError("No trainable NVFP4 weight quantizers found for compression proxy.")

    selected_refs = _sample_refs(refs, config, step)
    if config.proxy_type == "soft_code":
        return _soft_code_histogram_proxy(selected_refs, config, step)
    if config.proxy_type == "soft_byte":
        raise NotImplementedError("soft_byte proxy is not implemented yet.")
    raise ValueError(f"Unsupported compression proxy type: {config.proxy_type}")


def _is_static_nvfp4_weight_quantizer(quantizer: TensorQuantizer) -> bool:
    return (
        quantizer.is_enabled
        and quantizer.is_static_block_quant
        and quantizer.num_bits == (2, 1)
        and quantizer.block_sizes is not None
        and quantizer.block_sizes.get("scale_bits") == (4, 3)
        and quantizer.block_sizes.get(-1) == 16
        and hasattr(quantizer, "_amax")
    )


def _sample_refs(
    refs: list[NVFP4WeightRef], config: CompressionProxyConfig, step: int
) -> list[NVFP4WeightRef]:
    filtered = refs
    if config.layer_name_filter:
        filtered = [ref for ref in refs if config.layer_name_filter in ref.name]
        if not filtered:
            raise ValueError(
                f"No NVFP4 trainable weight quantizers matched filter '{config.layer_name_filter}'."
            )

    if config.sample_layers <= 0 or config.sample_layers >= len(filtered):
        return filtered

    rng = random.Random(config.seed + step)
    indices = list(range(len(filtered)))
    rng.shuffle(indices)
    return [filtered[idx] for idx in indices[: config.sample_layers]]


def _soft_code_histogram_proxy(
    refs: list[NVFP4WeightRef], config: CompressionProxyConfig, step: int
) -> CompressionProxyStats:
    device = refs[0].weight.device
    codebook = NVFP4QTensor.get_e2m1_values(device=device).to(torch.float32)
    histogram = torch.zeros(codebook.numel(), dtype=torch.float32, device=device)
    num_blocks_sampled = 0
    num_values_sampled = 0

    for idx, ref in enumerate(refs):
        blocks, scales = _get_sampled_blocks_and_scales(
            ref.weight, ref.quantizer, config.sample_blocks_per_layer, config.seed + step + idx
        )
        if blocks.numel() == 0:
            continue

        normalized = blocks / scales.unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        distances = normalized.unsqueeze(-1) - codebook.view(1, 1, -1)
        logits = -(distances.square()) / max(config.temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        histogram = histogram + probs.sum(dim=(0, 1))
        num_blocks_sampled += blocks.shape[0]
        num_values_sampled += blocks.numel()

    if num_values_sampled == 0:
        raise ValueError("Compression proxy sampled zero NVFP4 values.")

    histogram = histogram / float(num_values_sampled)
    entropy_bits = -(histogram * torch.log2(histogram.clamp_min(1e-12))).sum()
    return CompressionProxyStats(
        entropy_bits=entropy_bits,
        num_layers_sampled=len(refs),
        num_blocks_sampled=num_blocks_sampled,
        num_values_sampled=num_values_sampled,
    )


def _get_sampled_blocks_and_scales(
    weight: torch.Tensor,
    quantizer: TensorQuantizer,
    max_blocks: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = quantizer.block_sizes[-1]
    blocks = weight.detach() if not weight.requires_grad else weight
    blocks = blocks.to(torch.float32)
    if blocks.shape[-1] % block_size != 0:
        pad = block_size - (blocks.shape[-1] % block_size)
        blocks = F.pad(blocks, (0, pad))
    blocks = blocks.reshape(-1, block_size)

    per_block_amax = quantizer.amax
    if per_block_amax is None:
        raise ValueError("Static NVFP4 quantizer is missing amax.")
    per_block_amax = per_block_amax.to(blocks.device, dtype=torch.float32).reshape(-1)

    if blocks.shape[0] != per_block_amax.numel():
        raise ValueError(
            f"Weight block count {blocks.shape[0]} does not match per-block amax count {per_block_amax.numel()}."
        )

    scales = per_block_amax / 6.0
    global_amax = getattr(quantizer, "global_amax", None)
    if global_amax is not None:
        scale_quant_amax = global_amax.to(blocks.device, dtype=torch.float32) / 6.0
        scales = scaled_e4m3_impl(scales, scale_quant_amax)

    if max_blocks > 0 and blocks.shape[0] > max_blocks:
        generator = torch.Generator(device=blocks.device)
        generator.manual_seed(seed)
        perm = torch.randperm(blocks.shape[0], generator=generator, device=blocks.device)
        indices = perm[:max_blocks]
        blocks = blocks.index_select(0, indices)
        scales = scales.index_select(0, indices)

    return blocks, scales
