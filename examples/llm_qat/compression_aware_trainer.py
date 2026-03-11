# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental compression-aware QAT trainers for Hugging Face models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer, QATTrainer

from compression_proxy import (
    CompressionProxyConfig,
    CompressionProxyStats,
    collect_trainable_nvfp4_weight_refs,
    compute_compression_proxy,
)


@dataclass
class CompressionArguments:
    """Arguments for experimental compression-aware QAT."""

    compression_loss: bool = field(
        default=False,
        metadata={"help": "Enable experimental compression-aware regularization during QAT."},
    )
    compression_proxy_type: str = field(
        default="soft_code",
        metadata={"help": "Compression proxy type. soft_byte is reserved for a future version."},
    )
    compression_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for direct compression entropy regularization."},
    )
    compression_target_ratio: float = field(
        default=0.0,
        metadata={
            "help": "If > 0, set target entropy to this ratio times the post-quantization baseline entropy."
        },
    )
    compression_target_value: float = field(
        default=0.0,
        metadata={"help": "If > 0, use this absolute entropy target in bits instead of target_ratio."},
    )
    compression_penalty_rho: float = field(
        default=0.0,
        metadata={"help": "Quadratic penalty coefficient for constraint violation."},
    )
    compression_dual_lr: float = field(
        default=0.0,
        metadata={"help": "Dual update step size for the entropy constraint."},
    )
    compression_temperature: float = field(
        default=0.25,
        metadata={"help": "Softmax temperature for the soft code histogram proxy."},
    )
    compression_sample_layers: int = field(
        default=0,
        metadata={"help": "Number of quantized weight tensors sampled per step. 0 means all."},
    )
    compression_sample_blocks_per_layer: int = field(
        default=256,
        metadata={"help": "Maximum number of 16-weight blocks sampled from each tensor per step."},
    )
    compression_seed: int = field(
        default=1234,
        metadata={"help": "Base random seed for proxy sampling."},
    )
    compression_layer_name_filter: str = field(
        default="",
        metadata={"help": "Optional substring filter for quantized weight names."},
    )


class _CompressionAwareTrainerMixin:
    """Mixin implementing compression-aware regularization on top of ModelOpt QAT trainers."""

    def __init__(self, *args, compression_args: CompressionArguments | None = None, **kwargs):
        self.compression_args = compression_args or CompressionArguments()
        self._compression_weight_refs = None
        self._compression_target_value = None
        self._compression_dual = 0.0
        self._compression_last_metrics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _quantize_model(self):
        super()._quantize_model()
        self._initialize_compression_state(force=True)

    def _initialize_compression_state(self, force: bool = False):
        if not self.compression_args.compression_loss:
            return
        if self._compression_weight_refs is not None and not force:
            return

        model = self.model
        self._compression_weight_refs = collect_trainable_nvfp4_weight_refs(model)
        if not self._compression_weight_refs:
            raise RuntimeError(
                "Compression-aware QAT did not find any trainable static NVFP4 weight quantizers. "
                "Use a static NVFP4 weight config such as "
                "NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG or "
                "NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG."
            )

        target_override = self.compression_args.compression_target_value
        target_ratio = self.compression_args.compression_target_ratio
        if target_override > 0:
            self._compression_target_value = target_override
            return
        if target_ratio > 0:
            proxy_cfg = self._build_proxy_config(sample_all_layers=True)
            with torch.no_grad():
                stats = compute_compression_proxy(self._compression_weight_refs, proxy_cfg, step=0)
            self._compression_target_value = float(stats.entropy_bits.item()) * target_ratio
        else:
            self._compression_target_value = None

    def _build_proxy_config(self, sample_all_layers: bool = False) -> CompressionProxyConfig:
        sample_layers = 0 if sample_all_layers else self.compression_args.compression_sample_layers
        return CompressionProxyConfig(
            proxy_type=self.compression_args.compression_proxy_type,
            temperature=self.compression_args.compression_temperature,
            sample_layers=sample_layers,
            sample_blocks_per_layer=self.compression_args.compression_sample_blocks_per_layer,
            seed=self.compression_args.compression_seed,
            layer_name_filter=self.compression_args.compression_layer_name_filter,
        )

    def _compute_compression_regularizer(self) -> tuple[torch.Tensor, CompressionProxyStats]:
        self._initialize_compression_state()
        proxy_cfg = self._build_proxy_config()
        stats = compute_compression_proxy(
            self._compression_weight_refs,
            proxy_cfg,
            step=int(self.state.global_step),
        )

        entropy = stats.entropy_bits
        regularizer = entropy.new_zeros(())
        if self._compression_target_value is not None:
            target = entropy.new_tensor(self._compression_target_value)
            violation = entropy - target
            regularizer = regularizer + (self._compression_dual * violation)
            if self.compression_args.compression_penalty_rho > 0:
                regularizer = regularizer + (
                    0.5
                    * self.compression_args.compression_penalty_rho
                    * torch.relu(violation).square()
                )
            if self.model.training and self.compression_args.compression_dual_lr > 0:
                updated = self._compression_dual + (
                    self.compression_args.compression_dual_lr * float(violation.detach().item())
                )
                self._compression_dual = max(0.0, updated)
        else:
            regularizer = regularizer + (self.compression_args.compression_weight * entropy)

        self._compression_last_metrics = {
            "compression_entropy_bits": float(entropy.detach().item()),
            "compression_regularizer": float(regularizer.detach().item()),
            "compression_num_layers": float(stats.num_layers_sampled),
            "compression_num_blocks": float(stats.num_blocks_sampled),
            "compression_num_values": float(stats.num_values_sampled),
            "compression_dual": float(self._compression_dual),
        }
        if self._compression_target_value is not None:
            self._compression_last_metrics["compression_target_bits"] = float(
                self._compression_target_value
            )

        return regularizer, stats

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if self.compression_args.compression_loss and self.model.training:
            regularizer, _ = self._compute_compression_regularizer()
            loss = loss + regularizer

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._compression_last_metrics:
            logs = {**logs, **self._compression_last_metrics}
        super().log(logs, start_time=start_time)


class CompressionAwareQATTrainer(_CompressionAwareTrainerMixin, QATTrainer):
    """Experimental compression-aware QAT trainer."""


class CompressionAwareQADTrainer(_CompressionAwareTrainerMixin, QADTrainer):
    """Experimental compression-aware QAD trainer."""
