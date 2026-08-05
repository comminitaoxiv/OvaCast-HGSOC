from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ovacast.data.records import PatientRecord, PathwayDefinition


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    field: str
    message: str
    patient_key: str | None


@dataclass(frozen=True)
class CohortSummary:
    cohort: str
    patients: int
    events: int
    censored: int
    radiology_available: int
    pathology_available: int
    genes: int
    median_survival_months: float


@dataclass(frozen=True)
class ManifestEntry:
    logical_name: str
    relative_path: str
    bytes: int
    sha256: str


_ALLOWED_STAGES = {"I", "IA", "IB", "IC", "II", "IIA", "IIB", "IIC", "III", "IIIA", "IIIB", "IIIC", "IV"}
_ALLOWED_GRADES = {"G1", "G2", "G3", "G4"}
_ALLOWED_COHORTS = {"TCGA-OV", "TCIA-OV", "GSE26712", "GSE9891", "PTRC-HGSOC"}


def stable_patient_key(cohort: str, source_identifier: str) -> str:
    payload = f"{cohort}\x00{source_identifier}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def validate_patient(record: PatientRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    key = record.patient_id
    if record.cohort not in _ALLOWED_COHORTS:
        issues.append(ValidationIssue("error", "cohort", "unknown cohort", key))
    if len(record.patient_id) < 8:
        issues.append(ValidationIssue("error", "patient_id", "unstable patient key", key))
    if not record.molecular.expression:
        issues.append(ValidationIssue("error", "expression", "empty molecular profile", key))
    if record.clinical.age is not None and not 18 <= record.clinical.age <= 110:
        issues.append(ValidationIssue("error", "age", "age outside admissible range", key))
    stage = record.clinical.figo_stage
    if stage is not None and stage.upper() not in _ALLOWED_STAGES:
        issues.append(ValidationIssue("warning", "figo_stage", "unrecognized FIGO stage", key))
    grade = record.clinical.histological_grade
    if grade is not None and grade.upper() not in _ALLOWED_GRADES:
        issues.append(ValidationIssue("warning", "histological_grade", "unrecognized grade", key))
    if record.outcome is not None and record.outcome.time_months > 360:
        issues.append(ValidationIssue("warning", "survival_time", "survival exceeds thirty years", key))
    for gene, value in record.molecular.expression.items():
        if not gene or not np.isfinite(value):
            issues.append(ValidationIssue("error", "expression", "invalid gene value", key))
            break
    return issues


def validate_cohort(records: Sequence[PatientRecord]) -> list[ValidationIssue]:
    issues = [issue for record in records for issue in validate_patient(record)]
    identifiers = Counter(record.patient_id for record in records)
    for identifier, count in identifiers.items():
        if count > 1:
            issues.append(ValidationIssue("error", "patient_id", "duplicate patient key", identifier))
    cohorts = {record.cohort for record in records}
    if len(cohorts) > 1:
        issues.append(ValidationIssue("warning", "cohort", "mixed cohort collection", None))
    return issues


def require_valid_cohort(records: Sequence[PatientRecord]) -> None:
    errors = [issue for issue in validate_cohort(records) if issue.level == "error"]
    if errors:
        rendered = "; ".join(
            f"{issue.field}:{issue.message}:{issue.patient_key or '-'}"
            for issue in errors[:20]
        )
        raise ValueError(rendered)


def cohort_summary(records: Sequence[PatientRecord]) -> CohortSummary:
    if not records:
        raise ValueError("empty cohort")
    outcomes = [record.outcome for record in records if record.outcome is not None]
    times = [outcome.time_months for outcome in outcomes]
    events = sum(outcome.event.value for outcome in outcomes)
    genes = {gene for record in records for gene in record.molecular.expression}
    return CohortSummary(
        cohort=records[0].cohort,
        patients=len(records),
        events=events,
        censored=len(outcomes) - events,
        radiology_available=sum(record.radiology is not None for record in records),
        pathology_available=sum(bool(record.pathology_report) for record in records),
        genes=len(genes),
        median_survival_months=float(np.median(times)) if times else float("nan"),
    )


def pathway_issues(
    pathways: Sequence[PathwayDefinition],
    minimum_genes: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    identifiers = Counter(pathway.identifier for pathway in pathways)
    for pathway in pathways:
        if len(pathway.genes) < minimum_genes:
            issues.append(
                ValidationIssue(
                    "error",
                    "pathway",
                    f"{pathway.identifier} has fewer than {minimum_genes} genes",
                    None,
                )
            )
        if identifiers[pathway.identifier] > 1:
            issues.append(
                ValidationIssue("error", "pathway", "duplicate pathway identifier", None)
            )
        if len(set(pathway.genes)) != len(pathway.genes):
            issues.append(
                ValidationIssue("error", "pathway", "duplicate gene membership", None)
            )
    return issues


def cross_pathway_overlap(pathways: Sequence[PathwayDefinition]) -> dict[str, tuple[str, ...]]:
    membership: dict[str, list[str]] = {}
    for pathway in pathways:
        for gene in pathway.genes:
            membership.setdefault(gene, []).append(pathway.identifier)
    return {
        gene: tuple(identifiers)
        for gene, identifiers in membership.items()
        if len(identifiers) > 1
    }


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, paths: Iterable[Path]) -> list[ManifestEntry]:
    resolved_root = root.resolve()
    entries: list[ManifestEntry] = []
    for path in sorted(paths):
        resolved = path.resolve()
        relative = resolved.relative_to(resolved_root)
        entries.append(
            ManifestEntry(
                logical_name=path.stem,
                relative_path=relative.as_posix(),
                bytes=path.stat().st_size,
                sha256=sha256_stream(path),
            )
        )
    return entries


def write_manifest(path: Path, entries: Sequence[ManifestEntry]) -> None:
    payload = [asdict(entry) for entry in entries]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> list[ManifestEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("manifest")
    return [
        ManifestEntry(
            logical_name=str(item["logical_name"]),
            relative_path=str(item["relative_path"]),
            bytes=int(item["bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload
    ]


def verify_manifest(root: Path, entries: Sequence[ManifestEntry]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for entry in entries:
        path = root / entry.relative_path
        if not path.is_file():
            issues.append(ValidationIssue("error", entry.logical_name, "missing file", None))
            continue
        if path.stat().st_size != entry.bytes:
            issues.append(ValidationIssue("error", entry.logical_name, "size mismatch", None))
            continue
        if sha256_stream(path) != entry.sha256:
            issues.append(ValidationIssue("error", entry.logical_name, "digest mismatch", None))
    return issues


def compare_gene_spaces(
    training: Mapping[str, float],
    external: Mapping[str, float],
) -> tuple[int, int, float]:
    training_genes = set(training)
    external_genes = set(external)
    overlap = training_genes.intersection(external_genes)
    fraction = len(overlap) / len(training_genes) if training_genes else 0.0
    return len(training_genes), len(overlap), fraction


def require_gene_coverage(
    training: Mapping[str, float],
    external: Mapping[str, float],
    minimum_fraction: float,
) -> None:
    _, _, fraction = compare_gene_spaces(training, external)
    if fraction < minimum_fraction:
        raise ValueError(f"gene coverage {fraction:.3f}")


def event_distribution(records: Sequence[PatientRecord]) -> dict[str, float]:
    outcomes = [record.outcome for record in records if record.outcome is not None]
    if not outcomes:
        return {"observed": 0.0, "censored": 0.0}
    observed = sum(outcome.event.value for outcome in outcomes) / len(outcomes)
    return {"observed": observed, "censored": 1.0 - observed}


def modality_distribution(records: Sequence[PatientRecord]) -> dict[str, float]:
    denominator = max(len(records), 1)
    return {
        "genomics": sum(bool(record.molecular.expression) for record in records) / denominator,
        "pathology": sum(bool(record.pathology_report) for record in records) / denominator,
        "radiology": sum(record.radiology is not None for record in records) / denominator,
        "clinical": sum(record.clinical.age is not None for record in records) / denominator,
    }

