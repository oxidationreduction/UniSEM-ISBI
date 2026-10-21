"""Q/V-only low-rank branches used by the ISBI LightFT model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAQVConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha or dropout")


class LoRAQKVLinear(nn.Module):
    """Preserve the fused QKV projection and add branches to Q and V only."""

    def __init__(self, base: nn.Linear, config: LoRAQVConfig) -> None:
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError("Expected a SAM fused QKV linear projection")
        self.base = base
        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        for parameter in base.parameters():
            parameter.requires_grad = False
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_q_down = nn.Linear(base.in_features, self.rank, bias=False, **factory)
        self.lora_q_up = nn.Linear(self.rank, base.in_features, bias=False, **factory)
        self.lora_v_down = nn.Linear(base.in_features, self.rank, bias=False, **factory)
        self.lora_v_up = nn.Linear(self.rank, base.in_features, bias=False, **factory)
        nn.init.normal_(self.lora_q_down.weight, std=0.01)
        nn.init.normal_(self.lora_v_down.weight, std=0.01)
        nn.init.zeros_(self.lora_q_up.weight)
        nn.init.zeros_(self.lora_v_up.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(values)
        delta_q = self.lora_q_up(self.lora_q_down(dropped)) * self.scale
        delta_v = self.lora_v_up(self.lora_v_down(dropped)) * self.scale
        return self.base(values) + torch.cat((delta_q, torch.zeros_like(delta_q), delta_v), dim=-1)


def inject_lora_qv(encoder: nn.Module, config: LoRAQVConfig | None = None) -> dict[str, Any]:
    """Attach branches to every encoder block and freeze its original weights."""
    config = config or LoRAQVConfig()
    blocks = getattr(encoder, "blocks", None)
    if blocks is None or not len(blocks):
        raise ValueError("Q/V LoRA requires a SAM ViT encoder with attention blocks")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for block in blocks:
        qkv = block.attn.qkv
        if isinstance(qkv, LoRAQKVLinear):
            if qkv.rank != config.rank or qkv.alpha != config.alpha:
                raise ValueError("Encoder already contains a different LoRA configuration")
        else:
            block.attn.qkv = LoRAQKVLinear(qkv, config)
    names = []
    count = 0
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = ".lora_" in name
        if parameter.requires_grad:
            names.append(name)
            count += parameter.numel()
    encoder._unisem_lora_config = asdict(config)
    return {"blocks": len(blocks), "tensors": len(names), "trainable": int(count), **asdict(config)}


__all__ = ["LoRAQVConfig", "LoRAQKVLinear", "inject_lora_qv"]
