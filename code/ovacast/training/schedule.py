from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Phase(Enum):
    GENOMIC_PRETRAINING = "genomic_pretraining"
    MULTIMODAL_FINETUNING = "multimodal_finetuning"
    EXPLANATION_TUNING = "explanation_tuning"


@dataclass(frozen=True)
class PhaseSpecification:
    phase: Phase
    epochs: int
    include_radiology: bool
    include_pathology: bool
    include_explanation: bool
    modality_dropout: float


def curriculum(
    genomic_epochs: int = 5,
    multimodal_epochs: int = 10,
    explanation_epochs: int = 3,
    modality_dropout: float = 0.3,
) -> tuple[PhaseSpecification, ...]:
    return (
        PhaseSpecification(
            phase=Phase.GENOMIC_PRETRAINING,
            epochs=genomic_epochs,
            include_radiology=False,
            include_pathology=False,
            include_explanation=False,
            modality_dropout=0.0,
        ),
        PhaseSpecification(
            phase=Phase.MULTIMODAL_FINETUNING,
            epochs=multimodal_epochs,
            include_radiology=True,
            include_pathology=True,
            include_explanation=False,
            modality_dropout=modality_dropout,
        ),
        PhaseSpecification(
            phase=Phase.EXPLANATION_TUNING,
            epochs=explanation_epochs,
            include_radiology=True,
            include_pathology=True,
            include_explanation=True,
            modality_dropout=modality_dropout,
        ),
    )


def cosine_warmup_decay(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    minimum_ratio: float = 0.1,
) -> LambdaLR:
    warmup_steps = max(round(total_steps * warmup_ratio), 1)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, scale)


def phase_steps(
    sample_count: int,
    batch_size: int,
    gradient_accumulation: int,
    phases: tuple[PhaseSpecification, ...],
) -> dict[Phase, int]:
    updates_per_epoch = math.ceil(
        math.ceil(sample_count / batch_size) / gradient_accumulation
    )
    return {phase.phase: phase.epochs * updates_per_epoch for phase in phases}


def total_curriculum_steps(
    sample_count: int,
    batch_size: int,
    gradient_accumulation: int,
    phases: tuple[PhaseSpecification, ...],
) -> int:
    return sum(phase_steps(sample_count, batch_size, gradient_accumulation, phases).values())


def phase_for_epoch(
    epoch: int,
    phases: tuple[PhaseSpecification, ...],
) -> tuple[PhaseSpecification, int]:
    cursor = 0
    for phase in phases:
        if epoch < cursor + phase.epochs:
            return phase, epoch - cursor
        cursor += phase.epochs
    raise IndexError(epoch)

