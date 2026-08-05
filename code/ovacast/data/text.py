from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ovacast.data.records import ClinicalCovariates, RadiologyFeatures


_IMAGE_ABSENT = (
    "[IMAGE MODALITY ABSENT(CT): CT imaging unavailable. "
    "All predictions are based on genomic and textual narrative data only.]"
)


@dataclass(frozen=True)
class TextPolicy:
    maximum_report_characters: int = 24000
    collapse_whitespace: bool = True


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    return "".join(character for character in text if character.isprintable() or character == "\n")


def normalize_report(text: str, policy: TextPolicy = TextPolicy()) -> str:
    result = remove_control_characters(text)
    if policy.collapse_whitespace:
        result = normalize_whitespace(result)
    return result[: policy.maximum_report_characters]


def value_or_unknown(value: object | None) -> str:
    if value is None:
        return "not reported"
    return str(value)


def clinical_token(covariates: ClinicalCovariates) -> str:
    platinum = None
    if covariates.platinum_status is not None:
        platinum = covariates.platinum_status.name.lower()
    return (
        f"Age: {value_or_unknown(covariates.age)}. "
        f"FIGO stage: {value_or_unknown(covariates.figo_stage)}. "
        f"Histological grade: {value_or_unknown(covariates.histological_grade)}. "
        f"Debulking: {value_or_unknown(covariates.debulking)}. "
        f"Residual disease in centimeters: {value_or_unknown(covariates.residual_disease_cm)}. "
        f"Platinum response: {value_or_unknown(platinum)}."
    )


def radiology_token(features: RadiologyFeatures | None) -> str:
    if features is None:
        return _IMAGE_ABSENT
    values = (
        ("Primary ovarian mass", features.primary_mass),
        ("Peritoneal spread", features.peritoneal_spread),
        ("Mesenteric infiltration", features.mesenteric_infiltration),
        ("Other implant sites", features.other_implants),
        ("Pleural effusion", features.pleural_effusion),
        ("Ascites", features.ascites),
        ("Lymphadenopathy", features.lymphadenopathy),
        ("Distant metastases", features.distant_metastases),
    )
    return " ".join(f"{name}: {value_or_unknown(value)}." for name, value in values)


def pathology_token(report: str | None, policy: TextPolicy = TextPolicy()) -> str:
    if report is None or not report.strip():
        return "[PATHOLOGY REPORT ABSENT]"
    return normalize_report(report, policy)


def task_token(task: str) -> str:
    tasks = {
        "survival": "Predict five-year overall survival risk for this patient with HGSOC.",
        "subtype": "Classify the CLOVAR molecular subtype for this patient with HGSOC.",
        "platinum": "Predict platinum treatment sensitivity for this patient with HGSOC.",
        "explanation": "Explain the prognostic evidence across available modalities.",
    }
    if task not in tasks:
        raise KeyError(task)
    return tasks[task]


def join_sections(sections: Mapping[str, str]) -> str:
    return "\n\n".join(f"[{name.upper()}]\n{value}" for name, value in sections.items())


def multimodal_sequence(
    task: str,
    clinical: ClinicalCovariates,
    pathology_report: str | None,
    radiology: RadiologyFeatures | None,
    genomic_tokens: str,
) -> str:
    return join_sections(
        {
            "task": task_token(task),
            "clinical": clinical_token(clinical),
            "pathology": pathology_token(pathology_report),
            "radiology": radiology_token(radiology),
            "genomics": genomic_tokens,
        }
    )


def modality_spans(sequence: str) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    matches = list(re.finditer(r"\[([A-Z]+)\]\n", sequence))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sequence)
        spans[match.group(1).lower()] = (start, end)
    return spans


def truncate_sections(
    sections: Sequence[tuple[str, Sequence[int]]],
    limit: int,
) -> list[int]:
    if limit <= 0:
        raise ValueError("limit")
    result: list[int] = []
    remaining = limit
    for _, tokens in sections:
        accepted = list(tokens[:remaining])
        result.extend(accepted)
        remaining -= len(accepted)
        if remaining == 0:
            break
    return result
