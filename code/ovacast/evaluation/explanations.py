from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExplanationScore:
    driver_recovery: float
    faithfulness: float
    cross_modal_coherence: float
    cited_drivers: tuple[str, ...]


def normalize_gene_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9-]", "", value.upper())


def extract_gene_symbols(
    explanation: str,
    gene_vocabulary: Sequence[str],
) -> tuple[str, ...]:
    vocabulary = {normalize_gene_symbol(gene) for gene in gene_vocabulary}
    words = {normalize_gene_symbol(word) for word in re.findall(r"[A-Za-z0-9-]+", explanation)}
    return tuple(sorted(vocabulary.intersection(words)))


def driver_recovery(
    explanation: str,
    known_drivers: Sequence[str],
) -> float:
    drivers = {normalize_gene_symbol(gene) for gene in known_drivers}
    cited = set(extract_gene_symbols(explanation, known_drivers))
    return len(cited) / len(drivers) if drivers else float("nan")


def attribution_faithfulness(
    cited_genes: Sequence[str],
    attribution: Mapping[str, float],
    expression_deviation: Mapping[str, float],
) -> float:
    genes = [
        gene
        for gene in cited_genes
        if gene in attribution and gene in expression_deviation
    ]
    if len(genes) < 2:
        return float("nan")
    left = np.asarray([attribution[gene] for gene in genes], dtype=np.float64)
    right = np.asarray([abs(expression_deviation[gene]) for gene in genes], dtype=np.float64)
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def cross_modal_coherence(
    explanation: str,
    modality_terms: Mapping[str, Sequence[str]],
) -> float:
    normalized = explanation.lower()
    present = 0
    for terms in modality_terms.values():
        if any(term.lower() in normalized for term in terms):
            present += 1
    return present / len(modality_terms) if modality_terms else float("nan")


def score_explanation(
    explanation: str,
    known_drivers: Sequence[str],
    attribution: Mapping[str, float],
    expression_deviation: Mapping[str, float],
    modality_terms: Mapping[str, Sequence[str]],
) -> ExplanationScore:
    cited = extract_gene_symbols(explanation, known_drivers)
    return ExplanationScore(
        driver_recovery=driver_recovery(explanation, known_drivers),
        faithfulness=attribution_faithfulness(cited, attribution, expression_deviation),
        cross_modal_coherence=cross_modal_coherence(explanation, modality_terms),
        cited_drivers=cited,
    )


def likert_summary(ratings: Sequence[int]) -> tuple[float, float]:
    values = np.asarray(ratings, dtype=np.float64)
    if np.any((values < 1) | (values > 5)):
        raise ValueError("likert range")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def balanced_explanation_sample(
    risk: Sequence[float],
    subtype: Sequence[int],
    count: int,
    seed: int,
) -> np.ndarray:
    risks = np.asarray(risk, dtype=np.float64)
    subtypes = np.asarray(subtype, dtype=np.int64)
    high = risks > np.median(risks)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    combinations = [(level, group) for level in (False, True) for group in np.unique(subtypes)]
    per_group = max(count // len(combinations), 1)
    for level, group in combinations:
        candidates = np.flatnonzero((high == level) & (subtypes == group))
        take = min(per_group, len(candidates))
        if take:
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = count - len(selected)
    if remaining > 0:
        candidates = np.setdiff1d(np.arange(len(risks)), np.asarray(selected))
        selected.extend(rng.choice(candidates, size=remaining, replace=False).tolist())
    return np.asarray(selected[:count], dtype=np.int64)

