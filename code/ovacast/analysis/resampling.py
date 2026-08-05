from collections.abc import Callable

import numpy as np


def bootstrap_interval(
    metric: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    if not arrays or any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("bootstrap arrays must have equal nonzero lengths")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for sample in range(resamples):
        indices = rng.integers(0, len(arrays[0]), len(arrays[0]))
        estimates[sample] = metric(*(array[indices] for array in arrays))
    low, high = np.nanquantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high), estimates


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.maximum.accumulate(ranked * np.arange(len(ranked), 0, -1))
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output
