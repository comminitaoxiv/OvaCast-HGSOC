from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class EventStatus(Enum):
    CENSORED = 0
    OBSERVED = 1


class PlatinumStatus(Enum):
    REFRACTORY = 0
    SENSITIVE = 1


class ClovarSubtype(Enum):
    DIFFERENTIATED = 0
    IMMUNOREACTIVE = 1
    MESENCHYMAL = 2
    PROLIFERATIVE = 3


@dataclass(frozen=True)
class SurvivalOutcome:
    time_months: float
    event: EventStatus

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_months) or self.time_months <= 0:
            raise ValueError("time_months")


@dataclass(frozen=True)
class ClinicalCovariates:
    age: float | None
    figo_stage: str | None
    histological_grade: str | None
    debulking: str | None
    residual_disease_cm: float | None
    platinum_status: PlatinumStatus | None


@dataclass(frozen=True)
class RadiologyFeatures:
    primary_mass: str | None
    peritoneal_spread: str | None
    mesenteric_infiltration: str | None
    other_implants: str | None
    pleural_effusion: str | None
    ascites: str | None
    lymphadenopathy: str | None
    distant_metastases: str | None


@dataclass(frozen=True)
class MolecularProfile:
    expression: Mapping[str, float]
    mutations: Sequence[str]
    copy_number_alterations: Sequence[str]


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    molecular: MolecularProfile
    clinical: ClinicalCovariates
    radiology: RadiologyFeatures | None
    pathology_report: str | None
    outcome: SurvivalOutcome | None
    subtype: ClovarSubtype | None
    cohort: str


@dataclass(frozen=True)
class PathwayDefinition:
    identifier: str
    name: str
    source: str
    genes: tuple[str, ...]


@dataclass(frozen=True)
class PathwayStatistic:
    identifier: str
    name: str
    source: str
    activity: float
    heterogeneity: float
    notable_genes: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class NormalizationState:
    genes: tuple[str, ...]
    means: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]


@dataclass(frozen=True)
class EncodedPatient:
    patient_id: str
    sequence: str
    survival_time: float
    event: int
    cohort: str


def empty_radiology() -> RadiologyFeatures:
    return RadiologyFeatures(
        primary_mass=None,
        peritoneal_spread=None,
        mesenteric_infiltration=None,
        other_implants=None,
        pleural_effusion=None,
        ascites=None,
        lymphadenopathy=None,
        distant_metastases=None,
    )
