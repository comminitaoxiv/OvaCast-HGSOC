from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class Pathway:
    identifier: str
    name: str
    genes: tuple[str, ...]
    source: Literal["KEGG", "Reactome"]


@dataclass(frozen=True)
class ClinicalProfile:
    age: float
    stage: str
    grade: str
    debulking: str


@dataclass(frozen=True)
class RadiologyProfile:
    ovarian_mass: str
    peritoneal_spread: str
    mesenteric_infiltration: str
    implants: str
    effusion_ascites: str
    lymphadenopathy: str
    distant_metastases: str


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    expression: dict[str, float]
    survival_months: float
    event: int
    clinical: ClinicalProfile
    pathology: str
    radiology: RadiologyProfile | None = None
    mutations: tuple[str, ...] = ()
    copy_number: tuple[str, ...] = ()
    subtype: int | None = None
    platinum_response: int | None = None


@dataclass(frozen=True)
class NormalizationState:
    genes: tuple[str, ...]
    means: torch.Tensor
    standard_deviations: torch.Tensor


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    survival_months: torch.Tensor
    events: torch.Tensor
    explanation_labels: torch.Tensor | None
    patient_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelResult:
    risk: torch.Tensor
    language_logits: torch.Tensor | None
    hidden: torch.Tensor


@dataclass(frozen=True)
class Phase:
    name: str
    epochs: int
    explanation: bool


@dataclass
class RunState:
    phase: int = 0
    epoch: int = 0
    step: int = 0
    best_concordance: float = float("-inf")
    stale_epochs: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Paths:
    cohort: Path
    pathways: Path
    output: Path
