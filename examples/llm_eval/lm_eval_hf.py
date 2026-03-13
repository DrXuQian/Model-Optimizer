# Adapted from https://github.com/EleutherAI/lm-evaluation-harness
#
# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt-aware wrapper for lm-evaluation-harness HF models.

This script targets lm_eval==0.4.11, whose CLI is based on HarnessCLI/Run rather
than the older setup_parser/parse_eval_args helpers.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

from lm_eval.__main__ import cli_evaluate
from lm_eval._cli import harness as harness_cli
from lm_eval.models.huggingface import HFLM

import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized

_MODEL_OPT_ARGS = (
    "quant_cfg",
    "calib_batch_size",
    "calib_size",
    "auto_quantize_bits",
    "auto_quantize_method",
    "auto_quantize_score_size",
    "auto_quantize_checkpoint",
    "compress",
    "sparse_cfg",
)


def _import_quantize_model():
    from quantization_utils import quantize_model

    return quantize_model


def _import_sparsify_model():
    from sparse_attention_utils import sparsify_model

    return sparsify_model


def _is_attn_sparsified(model: Any) -> bool:
    try:
        from modelopt.torch.sparsity.attention_sparsity.conversion import is_attn_sparsified

        return bool(is_attn_sparsified(model))
    except ModuleNotFoundError:
        warnings.warn(
            "Sparse attention helpers are unavailable in this modelopt install; "
            "skipping is_attn_sparsified() check."
        )
        return False


def create_from_arg_obj(
    cls: type[HFLM], arg_dict: dict[str, Any], additional_config: dict | None = None
) -> HFLM:
    """Extend HFLM.create_from_arg_obj with ModelOpt quantization/sparsity hooks."""

    arg_dict = dict(arg_dict)
    quant_cfg = arg_dict.pop("quant_cfg", None)
    auto_quantize_bits = arg_dict.pop("auto_quantize_bits", None)
    auto_quantize_method = arg_dict.pop("auto_quantize_method", "gradient")
    auto_quantize_score_size = arg_dict.pop("auto_quantize_score_size", 128)
    auto_quantize_checkpoint = arg_dict.pop("auto_quantize_checkpoint", None)
    calib_batch_size = arg_dict.pop("calib_batch_size", None)
    calib_size = arg_dict.pop("calib_size", 512)
    compress = arg_dict.pop("compress", False)
    sparse_cfg = arg_dict.pop("sparse_cfg", None)

    additional_config = {} if additional_config is None else additional_config
    additional_config = {k: v for k, v in additional_config.items() if v is not None}

    # Enable automatic save/load of modelopt state huggingface checkpointing.
    mto.enable_huggingface_checkpointing()

    model_obj = cls(**arg_dict, **additional_config)
    model_obj.tokenizer.padding_side = "left"

    if is_quantized(model_obj.model):
        warnings.warn("Skipping quantization: model is already quantized.")
        return model_obj

    if quant_cfg:
        if not calib_batch_size:
            calib_batch_size = model_obj.batch_size

        quantize_model = _import_quantize_model()
        quantize_model(
            model=model_obj,
            quant_cfg=quant_cfg.split(",") if auto_quantize_bits is not None else quant_cfg,
            tokenizer=model_obj.tokenizer,
            batch_size=calib_batch_size,
            calib_size=calib_size,
            auto_quantize_bits=auto_quantize_bits,
            auto_quantize_method=auto_quantize_method,
            auto_quantize_score_size=auto_quantize_score_size,
            test_generated=False,
            compress=compress,
            auto_quantize_checkpoint=auto_quantize_checkpoint,
        )

    if sparse_cfg:
        if _is_attn_sparsified(model_obj.model):
            warnings.warn("Skipping sparse attention: model already has sparse attention applied.")
        else:
            sparsify_model = _import_sparsify_model()
            sparsify_model(
                model=model_obj,
                sparse_cfg=sparse_cfg,
            )

    return model_obj


def _patch_hflm() -> None:
    HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)


def _patch_run_parser() -> None:
    original_add_args = harness_cli.Run._add_args
    original_execute = harness_cli.Run._execute

    def _add_args(self) -> None:
        original_add_args(self)

        modelopt_group = self._parser.add_argument_group("modelopt")
        modelopt_group.add_argument(
            "--quant_cfg",
            type=str,
            help=(
                "Quantization format. If --auto_quantize_bits is specified, this argument specifies "
                "the comma-separated list of quantization formats searched by auto_quantize."
            ),
        )
        modelopt_group.add_argument(
            "--calib_batch_size",
            type=int,
            help="Batch size for quantization calibration",
        )
        modelopt_group.add_argument(
            "--calib_size",
            type=int,
            default=512,
            help="Calibration size for quantization",
        )
        modelopt_group.add_argument(
            "--auto_quantize_bits",
            type=float,
            help=(
                "Effective bits constraint for auto_quantize. If not set, regular quantization "
                "will be applied."
            ),
        )
        modelopt_group.add_argument(
            "--auto_quantize_method",
            type=str,
            default="gradient",
            choices=["gradient", "kl_div"],
            help=(
                "Method for auto_quantize sensitivity analysis. 'gradient' uses labels; 'kl_div' "
                "uses KL divergence between original and quantized outputs."
            ),
        )
        modelopt_group.add_argument(
            "--auto_quantize_score_size",
            type=int,
            default=128,
            help="Number of samples to use for auto_quantize scoring.",
        )
        modelopt_group.add_argument(
            "--auto_quantize_checkpoint",
            type=str,
            help="Path to checkpoint file for saving/restoring auto_quantize search state.",
        )
        modelopt_group.add_argument(
            "--compress",
            action="store_true",
            help="Compress the model after quantization.",
        )
        modelopt_group.add_argument(
            "--sparse_cfg",
            type=str,
            help="Sparse attention configuration (e.g. SKIP_SOFTMAX_DEFAULT).",
        )

    def _execute(args: argparse.Namespace) -> None:
        model_args = {} if args.model_args is None else dict(args.model_args)
        modelopt_values: dict[str, Any] = {}

        if getattr(args, "trust_remote_code", False):
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
            model_args["trust_remote_code"] = True
            args.trust_remote_code = None

        for key in _MODEL_OPT_ARGS:
            if hasattr(args, key):
                modelopt_values[key] = getattr(args, key)
                delattr(args, key)

        enable_modelopt = any(
            [
                modelopt_values.get("quant_cfg") is not None,
                modelopt_values.get("auto_quantize_bits") is not None,
                modelopt_values.get("auto_quantize_checkpoint") is not None,
                modelopt_values.get("calib_batch_size") is not None,
                bool(modelopt_values.get("compress")),
                modelopt_values.get("sparse_cfg") is not None,
            ]
        )

        if enable_modelopt:
            for key, value in modelopt_values.items():
                if value is None:
                    continue
                if key == "compress" and value is False:
                    continue
                model_args[key] = value

        args.model_args = model_args
        original_execute(args)

    harness_cli.Run._add_args = _add_args
    harness_cli.Run._execute = staticmethod(_execute)


def main() -> None:
    _patch_hflm()
    _patch_run_parser()
    cli_evaluate()


if __name__ == "__main__":
    main()
