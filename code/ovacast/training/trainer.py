from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from ovacast.losses.multitask import LossBreakdown, LossWeights, multimodal_loss
from ovacast.models.batching import ClinicalBatch, move_batch
from ovacast.models.ovacast import OvaCast
from ovacast.training.schedule import PhaseSpecification
from ovacast.training.state import TrainingState, atomic_save, checkpoint_payload


LOGGER = logging.getLogger("ovacast.training")


@dataclass(frozen=True)
class TrainerSettings:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    precision: str = "bf16"
    checkpoint_directory: Path = Path("artifacts")
    log_interval: int = 10


@dataclass
class RunningLoss:
    total: float = 0.0
    survival: float = 0.0
    explanation: float = 0.0
    subtype: float = 0.0
    platinum: float = 0.0
    batches: int = 0

    def update(self, loss: LossBreakdown) -> None:
        self.total += float(loss.total.detach())
        self.survival += float(loss.survival.detach())
        self.explanation += float(loss.explanation.detach())
        self.subtype += float(loss.subtype.detach())
        self.platinum += float(loss.platinum.detach())
        self.batches += 1

    def means(self) -> dict[str, float]:
        denominator = max(self.batches, 1)
        return {
            "total": self.total / denominator,
            "survival": self.survival / denominator,
            "explanation": self.explanation / denominator,
            "subtype": self.subtype / denominator,
            "platinum": self.platinum / denominator,
        }


class Trainer:
    def __init__(
        self,
        model: OvaCast,
        settings: TrainerSettings,
        device: torch.device,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.model = model.to(device)
        self.settings = settings
        self.device = device
        self.optimizer = optimizer or AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=settings.precision == "fp16")

    def autocast_context(self) -> Any:
        enabled = self.device.type == "cuda" and self.settings.precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.settings.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def forward_loss(
        self,
        batch: ClinicalBatch,
        phase: PhaseSpecification,
        weights: LossWeights,
    ) -> LossBreakdown:
        output = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.language_labels if phase.include_explanation else None,
        )
        return multimodal_loss(
            output,
            batch,
            weights,
            include_explanation=phase.include_explanation,
            include_subtype=True,
            include_platinum=True,
        )

    def backward(self, loss: Tensor) -> None:
        scaled = loss / self.settings.gradient_accumulation_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def optimizer_step(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(
        self,
        loader: DataLoader[ClinicalBatch],
        phase: PhaseSpecification,
        state: TrainingState,
        weights: LossWeights,
        scheduler: Any,
    ) -> dict[str, float]:
        self.model.train()
        running = RunningLoss()
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            moved = move_batch(batch, self.device)
            with self.autocast_context():
                loss = self.forward_loss(moved, phase, weights)
            self.backward(loss.total)
            running.update(loss)
            boundary = (batch_index + 1) % self.settings.gradient_accumulation_steps == 0
            final = batch_index + 1 == len(loader)
            if boundary or final:
                self.optimizer_step()
                scheduler.step()
                state.global_step += 1
            if state.global_step % self.settings.log_interval == 0:
                LOGGER.info("phase=%s step=%d losses=%s", phase.phase.value, state.global_step, running.means())
        return running.means()

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader[ClinicalBatch],
        phase: PhaseSpecification,
        weights: LossWeights,
    ) -> dict[str, float]:
        self.model.eval()
        running = RunningLoss()
        for batch in loader:
            moved = move_batch(batch, self.device)
            with self.autocast_context():
                loss = self.forward_loss(moved, phase, weights)
            running.update(loss)
        return running.means()

    def fit_phase(
        self,
        train_loader: DataLoader[ClinicalBatch],
        validation_loader: DataLoader[ClinicalBatch],
        phase: PhaseSpecification,
        state: TrainingState,
        weights: LossWeights,
        scheduler: Any,
    ) -> TrainingState:
        state.phase = phase.phase.value
        for _ in range(phase.epochs):
            training = self.train_epoch(train_loader, phase, state, weights, scheduler)
            validation = self.validate(validation_loader, phase, weights)
            state.epoch += 1
            LOGGER.info("epoch=%d train=%s validation=%s", state.epoch, training, validation)
            if validation["total"] < state.best_validation_loss:
                state.best_validation_loss = validation["total"]
                payload = checkpoint_payload(self.model, self.optimizer, scheduler, state)
                atomic_save(payload, self.settings.checkpoint_directory / "best.pt")
            payload = checkpoint_payload(self.model, self.optimizer, scheduler, state)
            atomic_save(payload, self.settings.checkpoint_directory / "latest.pt")
        return state
