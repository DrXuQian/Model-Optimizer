# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for experimental NVFP4 scale inflation helpers."""

import torch

from experimental.nvfp4_scale_inflation.scale_inflation import (
    BLOCK_SIZE,
    compute_nvfp4_compression_rate,
    dequantize_nvfp4_from_checkpoint,
    encode_actual_scales,
    optimize_scales_ste,
    pack_nvfp4_codes,
    quantize_to_nvfp4,
    unpack_nvfp4_codes,
)


def test_pack_unpack_roundtrip():
    codes = torch.tensor([[0, 1, 2, 3, 8, 9, 10, 15]], dtype=torch.uint8)
    packed = pack_nvfp4_codes(codes)
    unpacked = unpack_nvfp4_codes(packed)
    assert torch.equal(unpacked, codes)


def test_dequantize_and_requantize_roundtrip():
    scales = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor([2.0], dtype=torch.float32)
    codes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.uint8)
    packed = pack_nvfp4_codes(codes)

    weight, actual_scales = dequantize_nvfp4_from_checkpoint(packed, scales, global_scale)
    requant_packed, requant_weight, requant_codes = quantize_to_nvfp4(weight, actual_scales)

    assert torch.allclose(requant_weight, weight)
    # -0 may be canonicalized back to +0 during re-quantization.
    assert torch.equal((requant_codes == 8), torch.zeros_like(requant_codes, dtype=torch.bool))
    assert torch.equal(fp4_zero_mask(requant_codes), fp4_zero_mask(codes))


def fp4_zero_mask(codes: torch.Tensor) -> torch.Tensor:
    return (codes == 0) | (codes == 8)


def test_encode_actual_scales_produces_positive_fp8_scales():
    actual_scales = torch.tensor([[0.25, 0.5, 1.0, 2.0]], dtype=torch.float32)
    fp8_scale, global_scale = encode_actual_scales(actual_scales)
    decoded = fp8_scale.float() / global_scale

    assert fp8_scale.dtype == torch.float8_e4m3fn
    assert global_scale.shape == (1,)
    assert float(global_scale.item()) > 0.0
    assert torch.all(decoded > 0)


def test_optimize_scales_respects_lower_bound():
    weight = torch.linspace(-2.0, 2.0, BLOCK_SIZE * 2, dtype=torch.float32).view(2, BLOCK_SIZE)
    original_scales = torch.tensor([0.5, 0.6], dtype=torch.float32)
    initial_scales = original_scales * 2.0

    optimized = optimize_scales_ste(
        weight,
        original_scales,
        initial_scales,
        steps=4,
        lr=1e-2,
        device="cpu",
        block_chunk_size=2,
    )

    assert torch.all(optimized >= original_scales)


def test_compression_rate_shape():
    packed = torch.zeros(1024, dtype=torch.uint8).numpy()
    stats = compute_nvfp4_compression_rate(packed)

    assert 0.0 <= stats["compression_rate"] <= 1.0
    assert stats["n_chunks"] >= 1
