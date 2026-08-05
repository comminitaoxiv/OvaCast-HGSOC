from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class KaplanMeierCurve:
    timeline: NDArray[np.float64]
    survival: NDArray[np.float64]
    at_risk: NDArray[np.int64]
    events: NDArray[np.int64]


def kaplan_meier(time: ArrayLike, event: ArrayLike) -> KaplanMeierCurve:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    if len(times) != len(events):
        raise ValueError("kaplan meier dimensions")
    timeline = np.unique(times[events == 1])
    survival = np.ones(len(timeline), dtype=np.float64)
    at_risk = np.zeros(len(timeline), dtype=np.int64)
    event_counts = np.zeros(len(timeline), dtype=np.int64)
    value = 1.0
    for index, point in enumerate(timeline):
        at_risk[index] = int(np.sum(times >= point))
        event_counts[index] = int(np.sum((times == point) & (events == 1)))
        if at_risk[index] > 0:
            value *= 1.0 - event_counts[index] / at_risk[index]
        survival[index] = value
    return KaplanMeierCurve(timeline, survival, at_risk, event_counts)


def survival_at(curve: KaplanMeierCurve, points: ArrayLike) -> NDArray[np.float64]:
    query = np.asarray(points, dtype=np.float64)
    indices = np.searchsorted(curve.timeline, query, side="right") - 1
    values = np.ones(len(query), dtype=np.float64)
    valid = indices >= 0
    values[valid] = curve.survival[indices[valid]]
    return values


def median_survival(curve: KaplanMeierCurve) -> float:
    below = np.flatnonzero(curve.survival <= 0.5)
    if len(below) == 0:
        return float("inf")
    return float(curve.timeline[below[0]])


def cumulative_hazard(curve: KaplanMeierCurve) -> NDArray[np.float64]:
    return -np.log(np.clip(curve.survival, 1e-12, 1.0))


def greenwood_variance(curve: KaplanMeierCurve) -> NDArray[np.float64]:
    terms = np.zeros(len(curve.timeline), dtype=np.float64)
    valid = curve.at_risk > curve.events
    terms[valid] = curve.events[valid] / (
        curve.at_risk[valid] * (curve.at_risk[valid] - curve.events[valid])
    )
    return curve.survival**2 * np.cumsum(terms)


def confidence_band(
    curve: KaplanMeierCurve,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z = 1.959963984540054
    variance = greenwood_variance(curve)
    error = z * np.sqrt(variance)
    lower = np.clip(curve.survival - error, 0.0, 1.0)
    upper = np.clip(curve.survival + error, 0.0, 1.0)
    return lower, upper


def median_risk_groups(risk: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(risk, dtype=np.float64)
    threshold = float(np.median(values))
    return (values > threshold).astype(np.int64)


def group_curves(
    time: ArrayLike,
    event: ArrayLike,
    groups: ArrayLike,
) -> dict[int, KaplanMeierCurve]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    labels = np.asarray(groups, dtype=np.int64)
    return {
        int(group): kaplan_meier(times[labels == group], events[labels == group])
        for group in np.unique(labels)
    }

