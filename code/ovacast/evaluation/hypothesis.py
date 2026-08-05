from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float


def holm_bonferroni(p_values: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)[::-1]
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_rank, index in enumerate(order):
        rank = count - reverse_rank
        candidate = values[index] * count / rank
        running = min(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def logrank_test(
    time: ArrayLike,
    event: ArrayLike,
    group: ArrayLike,
) -> TestResult:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    groups = np.asarray(group, dtype=np.int64)
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError("logrank groups")
    observed_first = 0.0
    expected_first = 0.0
    variance = 0.0
    for point in np.unique(times[events == 1]):
        at_risk = times >= point
        failures = (times == point) & (events == 1)
        risk_total = int(np.sum(at_risk))
        event_total = int(np.sum(failures))
        risk_first = int(np.sum(at_risk & (groups == unique_groups[0])))
        event_first = int(np.sum(failures & (groups == unique_groups[0])))
        if risk_total <= 1:
            continue
        expected = event_total * risk_first / risk_total
        factor = (risk_total - event_total) / (risk_total - 1)
        variance_term = (
            risk_first
            * (risk_total - risk_first)
            * event_total
            * factor
            / (risk_total * risk_total)
        )
        observed_first += event_first
        expected_first += expected
        variance += variance_term
    statistic = (observed_first - expected_first) ** 2 / max(variance, 1e-12)
    return TestResult(float(statistic), float(stats.chi2.sf(statistic, 1)))


def permutation_test(
    first: ArrayLike,
    second: ArrayLike,
    repetitions: int = 10000,
    seed: int = 42,
) -> TestResult:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    observed = float(np.mean(left) - np.mean(right))
    combined = np.concatenate([left, right])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(repetitions):
        permuted = rng.permutation(combined)
        difference = np.mean(permuted[: len(left)]) - np.mean(permuted[len(left) :])
        count += int(abs(difference) >= abs(observed))
    return TestResult(observed, (count + 1) / (repetitions + 1))


def paired_permutation_test(
    first: ArrayLike,
    second: ArrayLike,
    repetitions: int = 10000,
    seed: int = 42,
) -> TestResult:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired dimensions")
    differences = left - right
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(repetitions, len(differences)))
    null = np.mean(signs * differences, axis=1)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (repetitions + 1)
    return TestResult(observed, float(p_value))


def cohen_kappa(first: ArrayLike, second: ArrayLike) -> float:
    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("rating dimensions")
    categories = np.union1d(left, right)
    observed = np.mean(left == right)
    expected = 0.0
    for category in categories:
        expected += np.mean(left == category) * np.mean(right == category)
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def spearman_correlation(first: ArrayLike, second: ArrayLike) -> TestResult:
    statistic, p_value = stats.spearmanr(first, second)
    return TestResult(float(statistic), float(p_value))


def kendall_correlation(first: ArrayLike, second: ArrayLike) -> TestResult:
    statistic, p_value = stats.kendalltau(first, second)
    return TestResult(float(statistic), float(p_value))

