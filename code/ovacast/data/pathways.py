from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ovacast.data.records import PathwayDefinition, PathwayStatistic


@dataclass(frozen=True)
class PathwayFilter:
    minimum_genes: int = 10
    require_nonoverlap: bool = True


def parse_gmt(lines: Iterable[str], source: str) -> list[PathwayDefinition]:
    pathways: list[PathwayDefinition] = []
    for raw in lines:
        fields = raw.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        name = fields[0].strip()
        identifier = fields[1].strip()
        genes = tuple(dict.fromkeys(gene.strip() for gene in fields[2:] if gene.strip()))
        pathways.append(PathwayDefinition(identifier, name, source, genes))
    return pathways


def filter_pathways(
    pathways: Sequence[PathwayDefinition],
    settings: PathwayFilter,
) -> list[PathwayDefinition]:
    eligible = [pathway for pathway in pathways if len(pathway.genes) >= settings.minimum_genes]
    eligible.sort(key=lambda item: (-len(item.genes), item.source, item.identifier))
    if not settings.require_nonoverlap:
        return eligible
    retained: list[PathwayDefinition] = []
    assigned: set[str] = set()
    for pathway in eligible:
        unique = tuple(gene for gene in pathway.genes if gene not in assigned)
        if len(unique) < settings.minimum_genes:
            continue
        retained.append(PathwayDefinition(pathway.identifier, pathway.name, pathway.source, unique))
        assigned.update(unique)
    return retained


def pathway_activity(
    standardized_expression: Mapping[str, float],
    pathway: PathwayDefinition,
) -> float:
    values = np.asarray(
        [
            standardized_expression[gene]
            for gene in pathway.genes
            if gene in standardized_expression
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def pathway_heterogeneity(
    standardized_expression: Mapping[str, float],
    pathway: PathwayDefinition,
) -> float:
    values = np.asarray(
        [
            standardized_expression[gene]
            for gene in pathway.genes
            if gene in standardized_expression
        ],
        dtype=np.float64,
    )
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def top_deviant_genes(
    standardized_expression: Mapping[str, float],
    pathway: PathwayDefinition,
    count: int = 3,
) -> tuple[tuple[str, float], ...]:
    available = (
        (gene, float(standardized_expression[gene]))
        for gene in pathway.genes
        if gene in standardized_expression
    )
    ordered = sorted(available, key=lambda item: (-abs(item[1]), item[0]))
    return tuple(ordered[:count])


def summarize_pathway(
    standardized_expression: Mapping[str, float],
    pathway: PathwayDefinition,
    top_count: int = 3,
) -> PathwayStatistic:
    return PathwayStatistic(
        identifier=pathway.identifier,
        name=pathway.name,
        source=pathway.source,
        activity=pathway_activity(standardized_expression, pathway),
        heterogeneity=pathway_heterogeneity(standardized_expression, pathway),
        notable_genes=top_deviant_genes(standardized_expression, pathway, top_count),
    )


def summarize_all(
    standardized_expression: Mapping[str, float],
    pathways: Sequence[PathwayDefinition],
    top_count: int = 3,
) -> list[PathwayStatistic]:
    return [summarize_pathway(standardized_expression, pathway, top_count) for pathway in pathways]


def activity_label(value: float) -> str:
    if value <= -2.0:
        return "very low"
    if value <= -1.0:
        return "low"
    if value <= -0.5:
        return "reduced"
    if value < 0.5:
        return "normal"
    if value < 1.0:
        return "moderately elevated"
    if value < 2.0:
        return "elevated"
    return "high"


def heterogeneity_label(value: float) -> str:
    if value < 0.5:
        return "low"
    if value < 1.0:
        return "moderate"
    return "high"


def gene_state(value: float) -> str:
    if value <= -2.0:
        return "very low"
    if value <= -0.75:
        return "reduced"
    if value < 0.75:
        return "normal"
    if value < 2.0:
        return "elevated"
    return "high"


def format_pathway_token(statistic: PathwayStatistic) -> str:
    notable = ", ".join(f"{gene} ({gene_state(value)})" for gene, value in statistic.notable_genes)
    return (
        f"{statistic.name} ({statistic.source} {statistic.identifier}): "
        f"activity = {activity_label(statistic.activity)} "
        f"(z = {statistic.activity:+.2f}), "
        f"heterogeneity = {heterogeneity_label(statistic.heterogeneity)} "
        f"(s = {statistic.heterogeneity:.2f}). "
        f"Notable genes: {notable}."
    )


def format_pathway_sequence(statistics: Sequence[PathwayStatistic]) -> str:
    return "\n".join(format_pathway_token(statistic) for statistic in statistics)


def mutation_tokens(mutations: Sequence[str]) -> str:
    if not mutations:
        return "No reportable somatic mutations detected."
    return "\n".join(f"Somatic alteration: {mutation.strip()}." for mutation in mutations)


def copy_number_tokens(alterations: Sequence[str]) -> str:
    if not alterations:
        return "No reportable copy number alterations detected."
    return "\n".join(f"Copy number alteration: {item.strip()}." for item in alterations)


def proteomic_proxy(
    protein_abundance: Mapping[str, float],
    pathways: Sequence[PathwayDefinition],
    top_count: int = 3,
) -> list[PathwayStatistic]:
    return summarize_all(protein_abundance, pathways, top_count)


def pathway_matrix(
    samples: Sequence[Mapping[str, float]],
    pathways: Sequence[PathwayDefinition],
) -> np.ndarray:
    matrix = np.zeros((len(samples), len(pathways)), dtype=np.float64)
    for row, sample in enumerate(samples):
        for column, pathway in enumerate(pathways):
            matrix[row, column] = pathway_activity(sample, pathway)
    return matrix


def pathway_coverage(
    sample: Mapping[str, float],
    pathways: Sequence[PathwayDefinition],
) -> dict[str, float]:
    coverage: dict[str, float] = {}
    genes = set(sample)
    for pathway in pathways:
        overlap = genes.intersection(pathway.genes)
        coverage[pathway.identifier] = len(overlap) / len(pathway.genes)
    return coverage


def require_coverage(
    sample: Mapping[str, float],
    pathways: Sequence[PathwayDefinition],
    minimum: float,
) -> list[PathwayDefinition]:
    coverage = pathway_coverage(sample, pathways)
    return [pathway for pathway in pathways if coverage[pathway.identifier] >= minimum]
