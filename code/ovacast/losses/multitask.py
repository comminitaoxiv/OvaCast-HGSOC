from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from ovacast.losses.survival import cox_partial_log_likelihood
from ovacast.models.batching import ClinicalBatch
from ovacast.models.ovacast import OvaCastOutput


@dataclass(frozen=True)
class LossWeights:
    survival: float = 1.0
    explanation: float = 0.1
    subtype: float = 1.0
    platinum: float = 1.0


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    survival: Tensor
    explanation: Tensor
    subtype: Tensor
    platinum: Tensor


def causal_language_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )


def optional_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    valid = targets >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[valid], targets[valid])


def optional_binary_loss(logits: Tensor, targets: Tensor) -> Tensor:
    valid = targets >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return functional.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid].to(dtype=logits.dtype),
    )


def multimodal_loss(
    output: OvaCastOutput,
    batch: ClinicalBatch,
    weights: LossWeights,
    include_explanation: bool,
    include_subtype: bool,
    include_platinum: bool,
) -> LossBreakdown:
    survival = cox_partial_log_likelihood(
        output.heads.log_hazard,
        batch.survival_time,
        batch.event,
    )
    explanation = (
        causal_language_loss(output.language_logits, batch.language_labels)
        if include_explanation
        else output.language_logits.sum() * 0.0
    )
    subtype = (
        optional_classification_loss(output.heads.subtype_logits, batch.subtype)
        if include_subtype
        else output.heads.subtype_logits.sum() * 0.0
    )
    platinum = (
        optional_binary_loss(output.heads.platinum_logit, batch.platinum)
        if include_platinum
        else output.heads.platinum_logit.sum() * 0.0
    )
    total = (
        weights.survival * survival
        + weights.explanation * explanation
        + weights.subtype * subtype
        + weights.platinum * platinum
    )
    return LossBreakdown(total, survival, explanation, subtype, platinum)
