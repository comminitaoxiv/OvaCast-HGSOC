from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ModalitySelection:
    clinical: bool
    pathology: bool
    radiology: bool
    genomics: bool


def sample_modality_selection(
    rng: np.random.Generator,
    dropout: float,
    available: Mapping[str, bool],
) -> ModalitySelection:
    if dropout < 0 or dropout > 1:
        raise ValueError("dropout")
    clinical = bool(available.get("clinical", True))
    pathology = bool(available.get("pathology", False)) and rng.random() >= dropout
    radiology = bool(available.get("radiology", False)) and rng.random() >= dropout
    genomics = bool(available.get("genomics", True)) and rng.random() >= dropout
    if not any((clinical, pathology, radiology, genomics)):
        genomics = bool(available.get("genomics", False))
        clinical = not genomics
    return ModalitySelection(clinical, pathology, radiology, genomics)


def apply_modality_selection(
    sections: Mapping[str, str],
    selection: ModalitySelection,
) -> dict[str, str]:
    absent = {
        "clinical": "[CLINICAL COVARIATES ABSENT]",
        "pathology": "[PATHOLOGY REPORT ABSENT]",
        "radiology": (
            "[IMAGE MODALITY ABSENT(CT): CT imaging unavailable. "
            "All predictions are based on genomic and textual narrative data only.]"
        ),
        "genomics": "[GENOMIC PROFILE ABSENT]",
    }
    enabled = {
        "clinical": selection.clinical,
        "pathology": selection.pathology,
        "radiology": selection.radiology,
        "genomics": selection.genomics,
    }
    return {
        modality: value if enabled[modality] else absent[modality]
        for modality, value in sections.items()
    }


def all_available() -> ModalitySelection:
    return ModalitySelection(True, True, True, True)


def genomics_only() -> ModalitySelection:
    return ModalitySelection(False, False, False, True)


def genomics_and_text() -> ModalitySelection:
    return ModalitySelection(True, True, False, True)


def without_pathway_tokenization() -> ModalitySelection:
    return ModalitySelection(True, True, True, False)


def selection_name(selection: ModalitySelection) -> str:
    active = [
        name
        for name, enabled in (
            ("clinical", selection.clinical),
            ("pathology", selection.pathology),
            ("radiology", selection.radiology),
            ("genomics", selection.genomics),
        )
        if enabled
    ]
    return "+".join(active)

