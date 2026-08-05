from dataclasses import dataclass
from math import sqrt

from ovacast.records.types import Pathway


@dataclass(frozen=True)
class PathwaySummary:
    pathway: Pathway
    activity: float
    heterogeneity: float
    notable_genes: tuple[tuple[str, float], ...]


def activity_label(value: float) -> str:
    if value <= -1.5:
        return "low"
    if value <= -0.5:
        return "reduced"
    if value < 0.5:
        return "normal"
    if value < 1.5:
        return "elevated"
    return "high"


def heterogeneity_label(value: float) -> str:
    if value < 0.7:
        return "low"
    if value < 1.2:
        return "moderate"
    return "high"


def summarize(pathway: Pathway, zscores: dict[str, float]) -> PathwaySummary:
    values = [(gene, zscores[gene]) for gene in pathway.genes if gene in zscores]
    if not values:
        return PathwaySummary(pathway, 0.0, 0.0, ())
    activity = sum(value for _, value in values) / len(values)
    variance = sum((value - activity) ** 2 for _, value in values) / max(len(values) - 1, 1)
    notable = tuple(sorted(values, key=lambda item: abs(item[1]), reverse=True)[:3])
    return PathwaySummary(pathway, activity, sqrt(variance), notable)


def render(summary: PathwaySummary) -> str:
    notable = ", ".join(
        f"{gene} ({activity_label(value)})" for gene, value in summary.notable_genes
    )
    return f"{summary.pathway.name} ({summary.pathway.source} {summary.pathway.identifier}): Activity = {activity_label(summary.activity)} (z = {summary.activity:+.2f}), Heterogeneity = {heterogeneity_label(summary.heterogeneity)} (s = {summary.heterogeneity:.2f}). Notable Genes: {notable}."


def tokenize_pathways(pathways: tuple[Pathway, ...], zscores: dict[str, float]) -> tuple[str, ...]:
    return tuple(render(summarize(pathway, zscores)) for pathway in pathways)


def mutation_tokens(mutations: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"Somatic alteration: {mutation}." for mutation in mutations)


def copy_number_tokens(copy_number: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"Copy-number alteration: {alteration}." for alteration in copy_number)
