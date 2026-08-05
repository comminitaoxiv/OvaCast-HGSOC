from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Metric = Callable[[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]], float]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float
    samples: NDArray[np.float64]


def bootstrap_interval(
    time: ArrayLike,
    event: ArrayLike,
    prediction: ArrayLike,
    metric: Metric,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    predictions = np.asarray(prediction, dtype=np.float64)
    if not (len(times) == len(events) == len(predictions)):
        raise ValueError("bootstrap dimensions")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(times), size=len(times))
        sampled_event = events[indices]
        if np.unique(sampled_event).size < 2:
            continue
        value = metric(times[indices], sampled_event, predictions[indices])
        if np.isfinite(value):
            values.append(value)
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("bootstrap samples")
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(metric(times, events, predictions)),
        lower=float(np.quantile(samples, tail)),
        upper=float(np.quantile(samples, 1.0 - tail)),
        standard_error=float(np.std(samples, ddof=1)),
        samples=samples,
    )


def paired_bootstrap_difference(
    time: ArrayLike,
    event: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    metric: Metric,
    repetitions: int = 2000,
    seed: int = 42,
) -> BootstrapInterval:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(times), size=len(times))
        if np.unique(events[indices]).size < 2:
            continue
        left = metric(times[indices], events[indices], first_values[indices])
        right = metric(times[indices], events[indices], second_values[indices])
        if np.isfinite(left) and np.isfinite(right):
            differences.append(left - right)
    samples = np.asarray(differences, dtype=np.float64)
    estimate = metric(times, events, first_values) - metric(times, events, second_values)
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        standard_error=float(np.std(samples, ddof=1)),
        samples=samples,
    )


def bootstrap_p_value(samples: ArrayLike, null: float = 0.0) -> float:
    values = np.asarray(samples, dtype=np.float64)
    left = np.mean(values <= null)
    right = np.mean(values >= null)
    return float(min(1.0, 2.0 * min(left, right)))

