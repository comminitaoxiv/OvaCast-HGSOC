from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ovacast.data.records import NormalizationState


def ordered_gene_union(samples: Sequence[Mapping[str, float]]) -> tuple[str, ...]:
    return tuple(sorted({gene for sample in samples for gene in sample}))


def expression_matrix(
    samples: Sequence[Mapping[str, float]],
    genes: Sequence[str],
    missing_value: float = np.nan,
) -> NDArray[np.float64]:
    matrix = np.full((len(samples), len(genes)), missing_value, dtype=np.float64)
    positions = {gene: index for index, gene in enumerate(genes)}
    for row, sample in enumerate(samples):
        for gene, value in sample.items():
            column = positions.get(gene)
            if column is not None:
                matrix[row, column] = float(value)
    return matrix


def fit_normalization(
    samples: Sequence[Mapping[str, float]],
    minimum_standard_deviation: float = 1e-8,
) -> NormalizationState:
    genes = ordered_gene_union(samples)
    matrix = expression_matrix(samples, genes)
    means = np.nanmean(matrix, axis=0)
    deviations = np.nanstd(matrix, axis=0, ddof=1)
    means = np.where(np.isfinite(means), means, 0.0)
    deviations = np.where(
        np.isfinite(deviations) & (deviations >= minimum_standard_deviation),
        deviations,
        1.0,
    )
    return NormalizationState(
        genes=genes,
        means=means.astype(np.float64),
        standard_deviations=deviations.astype(np.float64),
    )


def transform_expression(
    sample: Mapping[str, float],
    state: NormalizationState,
) -> dict[str, float]:
    values = np.asarray([sample.get(gene, np.nan) for gene in state.genes], dtype=np.float64)
    missing = ~np.isfinite(values)
    values[missing] = state.means[missing]
    standardized = (values - state.means) / state.standard_deviations
    return {gene: float(value) for gene, value in zip(state.genes, standardized, strict=True)}


def transform_many(
    samples: Sequence[Mapping[str, float]],
    state: NormalizationState,
) -> list[dict[str, float]]:
    return [transform_expression(sample, state) for sample in samples]


def serialize_state(state: NormalizationState) -> dict[str, object]:
    return {
        "genes": list(state.genes),
        "means": state.means.tolist(),
        "standard_deviations": state.standard_deviations.tolist(),
    }


def deserialize_state(payload: Mapping[str, object]) -> NormalizationState:
    genes_raw = payload["genes"]
    means_raw = payload["means"]
    deviations_raw = payload["standard_deviations"]
    if not isinstance(genes_raw, list) or not all(isinstance(item, str) for item in genes_raw):
        raise TypeError("genes")
    if not isinstance(means_raw, list) or not isinstance(deviations_raw, list):
        raise TypeError("normalization vectors")
    means = np.asarray(means_raw, dtype=np.float64)
    deviations = np.asarray(deviations_raw, dtype=np.float64)
    if len(genes_raw) != len(means) or len(means) != len(deviations):
        raise ValueError("normalization dimensions")
    return NormalizationState(tuple(genes_raw), means, deviations)


def quantile_normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if matrix.ndim != 2:
        raise ValueError("matrix")
    order = np.argsort(matrix, axis=0)
    sorted_values = np.sort(matrix, axis=0)
    reference = np.mean(sorted_values, axis=1)
    result = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        result[order[:, column], column] = reference
    return result


def select_highest_iqr_probe(
    probe_matrix: NDArray[np.float64],
    probe_ids: Sequence[str],
    probe_to_gene: Mapping[str, str],
) -> dict[str, int]:
    if probe_matrix.shape[1] != len(probe_ids):
        raise ValueError("probe identifiers")
    q75 = np.nanpercentile(probe_matrix, 75, axis=0)
    q25 = np.nanpercentile(probe_matrix, 25, axis=0)
    iqrs = q75 - q25
    selected: dict[str, int] = {}
    for index, probe in enumerate(probe_ids):
        gene = probe_to_gene.get(probe)
        if gene is None or not gene:
            continue
        previous = selected.get(gene)
        if previous is None or iqrs[index] > iqrs[previous]:
            selected[gene] = index
    return selected


def collapse_probes(
    probe_matrix: NDArray[np.float64],
    probe_ids: Sequence[str],
    probe_to_gene: Mapping[str, str],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    selected = select_highest_iqr_probe(probe_matrix, probe_ids, probe_to_gene)
    genes = tuple(sorted(selected))
    indices = [selected[gene] for gene in genes]
    return genes, probe_matrix[:, indices].astype(np.float64, copy=True)
