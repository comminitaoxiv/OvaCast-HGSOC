from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HeadOutput:
    log_hazard: Tensor
    subtype_logits: Tensor
    platinum_logit: Tensor


class SurvivalHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden).squeeze(-1)


class SubtypeHead(nn.Module):
    def __init__(self, hidden_size: int, classes: int = 4) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, classes)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.projection(self.normalization(hidden))


class PlatinumHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.projection(self.normalization(hidden)).squeeze(-1)


class ClinicalHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.survival = SurvivalHead(hidden_size)
        self.subtype = SubtypeHead(hidden_size)
        self.platinum = PlatinumHead(hidden_size)

    def forward(self, hidden: Tensor) -> HeadOutput:
        return HeadOutput(
            log_hazard=self.survival(hidden),
            subtype_logits=self.subtype(hidden),
            platinum_logit=self.platinum(hidden),
        )


def last_valid_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden state dimensions")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("attention dimensions")
    positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, positions]


def masked_mean_hidden(hidden: Tensor, mask: Tensor) -> Tensor:
    if hidden.shape[:2] != mask.shape:
        raise ValueError("mask dimensions")
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator
