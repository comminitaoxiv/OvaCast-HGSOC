from collections import defaultdict
from statistics import quantiles


def interquartile_range(values: list[float]) -> float:
    if len(values) < 4:
        return max(values, default=0.0) - min(values, default=0.0)
    points = quantiles(values, n=4, method="inclusive")
    return points[2] - points[0]


def choose_probe_per_gene(
    matrix: dict[str, list[float]], annotation: dict[str, str]
) -> dict[str, str]:
    candidates = defaultdict(list)
    for probe, gene in annotation.items():
        if probe in matrix and gene:
            candidates[gene.upper()].append(probe)
    return {
        gene: max(probes, key=lambda probe: interquartile_range(matrix[probe]))
        for gene, probes in candidates.items()
    }


def collapse_probes(
    matrix: dict[str, list[float]], annotation: dict[str, str]
) -> dict[str, list[float]]:
    selected = choose_probe_per_gene(matrix, annotation)
    return {gene: matrix[probe] for gene, probe in selected.items()}


def proteins_to_genes(abundance: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    grouped = defaultdict(list)
    for protein, value in abundance.items():
        gene = mapping.get(protein)
        if gene:
            grouped[gene.upper()].append(value)
    return {gene: sum(values) / len(values) for gene, values in grouped.items()}
