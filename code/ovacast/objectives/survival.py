import torch
from torch import nn


def cox_partial_log_likelihood(
    risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    if risk.ndim != 1 or times.ndim != 1 or events.ndim != 1:
        raise ValueError("risk, times, and events must be vectors")
    if not (risk.shape == times.shape == events.shape):
        raise ValueError("risk, times, and events must have equal shapes")
    order = torch.argsort(times, descending=True, stable=True)
    ordered_risk = risk[order]
    ordered_events = events[order].to(risk.dtype)
    log_risk_sum = torch.logcumsumexp(ordered_risk, dim=0)
    contributions = (ordered_risk - log_risk_sum) * ordered_events
    observed = ordered_events.sum().clamp_min(1.0)
    return -contributions.sum() / observed


class JointObjective(nn.Module):
    def __init__(self, explanation_weight: float = 0.1) -> None:
        super().__init__()
        self.explanation_weight = explanation_weight

    def forward(
        self,
        risk: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        language_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        survival = cox_partial_log_likelihood(risk, times, events)
        language = torch.zeros((), device=risk.device)
        if language_logits is not None and labels is not None:
            shifted_logits = language_logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            language = nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        total = survival + self.explanation_weight * language
        return total, {
            "total": total.detach(),
            "cox": survival.detach(),
            "language": language.detach(),
        }
