import logging
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import nn

from ovacast.objectives.survival import JointObjective
from ovacast.records.types import Batch


LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        accumulation: int = 8,
        explanation_weight: float = 0.1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accumulation = accumulation
        self.objective = JointObjective(explanation_weight)

    def train_epoch(self, batches: Iterable[Batch], explanation: bool) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "cox": 0.0, "language": 0.0}
        count = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches, start=1):
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if batch.input_ids.is_cuda
                else nullcontext()
            )
            with context:
                result = self.model(
                    batch.input_ids,
                    batch.attention_mask,
                    batch.explanation_labels if explanation else None,
                )
                loss, parts = self.objective(
                    result.risk,
                    batch.survival_months,
                    batch.events,
                    result.language_logits,
                    batch.explanation_labels if explanation else None,
                )
                scaled = loss / self.accumulation
            scaled.backward()
            if step % self.accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            for name, value in parts.items():
                totals[name] += float(value)
            count += 1
        if count % self.accumulation:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
        return {name: value / max(count, 1) for name, value in totals.items()}
