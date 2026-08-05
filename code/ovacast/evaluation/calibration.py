from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ovacast.metrics.survival_curves import kaplan_meier, survival_at


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    predicted: float
    observed: float
    count: int


def brier_score(
    time: ArrayLike,
    event: ArrayLike,
    survival_probability: ArrayLike,
    horizon: float,
) -> float:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    predictions = np.asarray(survival_probability, dtype=np.float64)
    outcome = (times > horizon).astype(np.float64)
    known = (times > horizon) | (events == 1)
    if not np.any(known):
        return float("nan")
    return float(np.mean((outcome[known] - predictions[known]) ** 2))


def integrated_brier_score(
    time: ArrayLike,
    event: ArrayLike,
    survival_matrix: ArrayLike,
    horizons: ArrayLike,
) -> float:
    grid = np.asarray(horizons, dtype=np.float64)
    matrix = np.asarray(survival_matrix, dtype=np.float64)
    if matrix.shape != (len(np.asarray(time)), len(grid)):
        raise ValueError("survival matrix")
    scores = np.asarray(
        [
            brier_score(time, event, matrix[:, column], horizon)
            for column, horizon in enumerate(grid)
        ],
        dtype=np.float64,
    )
    return float(np.trapz(scores, grid) / (grid[-1] - grid[0]))


def calibration_bins(
    time: ArrayLike,
    event: ArrayLike,
    predicted_survival: ArrayLike,
    horizon: float,
    bins: int = 5,
) -> list[CalibrationBin]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    predictions = np.asarray(predicted_survival, dtype=np.float64)
    edges = np.quantile(predictions, np.linspace(0, 1, bins + 1))
    assignments = np.digitize(predictions, edges[1:-1], right=True)
    result: list[CalibrationBin] = []
    for index in range(bins):
        selected = assignments == index
        if not np.any(selected):
            continue
        curve = kaplan_meier(times[selected], events[selected])
        observed = float(survival_at(curve, [horizon])[0])
        result.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                predicted=float(np.mean(predictions[selected])),
                observed=observed,
                count=int(np.sum(selected)),
            )
        )
    return result


def calibration_slope(
    observed: ArrayLike,
    predicted: ArrayLike,
) -> tuple[float, float]:
    targets = np.asarray(observed, dtype=np.float64)
    probabilities = np.asarray(predicted, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    design = np.column_stack([np.ones(len(logits)), logits])
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def expected_calibration_error(bins: list[CalibrationBin]) -> float:
    total = sum(item.count for item in bins)
    if total == 0:
        return float("nan")
    return float(
        sum(item.count * abs(item.predicted - item.observed) for item in bins) / total
    )


def baseline_survival(
    time: ArrayLike,
    event: ArrayLike,
    log_hazard: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    risks = np.exp(np.asarray(log_hazard, dtype=np.float64))
    event_times = np.unique(times[events == 1])
    increments = np.zeros(len(event_times), dtype=np.float64)
    for index, point in enumerate(event_times):
        failures = np.sum((times == point) & (events == 1))
        denominator = np.sum(risks[times >= point])
        increments[index] = failures / max(denominator, 1e-12)
    cumulative = np.cumsum(increments)
    return event_times, np.exp(-cumulative)


def predict_survival(
    baseline_times: ArrayLike,
    baseline_values: ArrayLike,
    log_hazard: ArrayLike,
    horizons: ArrayLike,
) -> NDArray[np.float64]:
    timeline = np.asarray(baseline_times, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    risk = np.exp(np.asarray(log_hazard, dtype=np.float64))
    grid = np.asarray(horizons, dtype=np.float64)
    positions = np.searchsorted(timeline, grid, side="right") - 1
    selected = np.ones(len(grid), dtype=np.float64)
    valid = positions >= 0
    selected[valid] = baseline[positions[valid]]
    return selected[None, :] ** risk[:, None]

