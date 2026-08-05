from __future__ import annotations

import torch
from torch import Tensor


def cox_partial_log_likelihood(
    log_hazard: Tensor,
    time: Tensor,
    event: Tensor,
    reduction: str = "mean",
) -> Tensor:
    if log_hazard.ndim != 1 or time.ndim != 1 or event.ndim != 1:
        raise ValueError("cox dimensions")
    if not (len(log_hazard) == len(time) == len(event)):
        raise ValueError("cox lengths")
    order = torch.argsort(time, descending=True, stable=True)
    ordered_hazard = log_hazard[order]
    ordered_event = event[order].to(dtype=log_hazard.dtype)
    log_risk = torch.logcumsumexp(ordered_hazard, dim=0)
    contributions = (ordered_hazard - log_risk) * ordered_event
    observed = ordered_event.sum().clamp_min(1.0)
    negative = -contributions.sum()
    if reduction == "sum":
        return negative
    if reduction == "mean":
        return negative / observed
    if reduction == "none":
        return -contributions
    raise ValueError(reduction)


def efron_cox_loss(
    log_hazard: Tensor,
    time: Tensor,
    event: Tensor,
) -> Tensor:
    unique_times = torch.unique(time[event > 0])
    total = log_hazard.new_zeros(())
    events_total = log_hazard.new_zeros(())
    risks = torch.exp(log_hazard)
    for event_time in unique_times:
        tied = (time == event_time) & (event > 0)
        risk_set = time >= event_time
        count = tied.sum()
        tied_risk = risks[tied].sum()
        risk_sum = risks[risk_set].sum()
        total = total - log_hazard[tied].sum()
        for offset in range(int(count.item())):
            fraction = offset / max(int(count.item()), 1)
            total = total + torch.log(risk_sum - fraction * tied_risk)
        events_total = events_total + count
    return total / events_total.clamp_min(1)


def pairwise_ranking_loss(
    log_hazard: Tensor,
    time: Tensor,
    event: Tensor,
    margin: float = 0.0,
) -> Tensor:
    earlier = time[:, None] < time[None, :]
    comparable = earlier & event[:, None].bool()
    differences = log_hazard[:, None] - log_hazard[None, :]
    losses = torch.relu(margin - differences)
    selected = losses[comparable]
    if selected.numel() == 0:
        return log_hazard.sum() * 0.0
    return selected.mean()


def negative_log_likelihood_discrete(
    logits: Tensor,
    event_bin: Tensor,
    event: Tensor,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("discrete survival dimensions")
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards + 1e-7, dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    bins = event_bin.long().clamp(0, logits.shape[1] - 1)
    prior = torch.where(
        bins > 0,
        survival[rows, (bins - 1).clamp_min(0)],
        torch.ones_like(hazards[:, 0]),
    )
    event_probability = prior * hazards[rows, bins]
    censor_probability = survival[rows, bins]
    probability = torch.where(event.bool(), event_probability, censor_probability)
    return -torch.log(probability.clamp_min(1e-7)).mean()
