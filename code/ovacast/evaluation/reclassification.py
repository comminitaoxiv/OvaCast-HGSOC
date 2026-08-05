from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ReclassificationResult:
    event_improvement: float
    nonevent_improvement: float
    net_reclassification: float


def category_free_nri(
    outcome: ArrayLike,
    reference_risk: ArrayLike,
    candidate_risk: ArrayLike,
) -> ReclassificationResult:
    outcomes = np.asarray(outcome, dtype=np.int64)
    reference = np.asarray(reference_risk, dtype=np.float64)
    candidate = np.asarray(candidate_risk, dtype=np.float64)
    if not (outcomes.shape == reference.shape == candidate.shape):
        raise ValueError("reclassification dimensions")
    events = outcomes == 1
    nonevents = outcomes == 0
    event_up = np.mean(candidate[events] > reference[events])
    event_down = np.mean(candidate[events] < reference[events])
    nonevent_down = np.mean(candidate[nonevents] < reference[nonevents])
    nonevent_up = np.mean(candidate[nonevents] > reference[nonevents])
    event_improvement = event_up - event_down
    nonevent_improvement = nonevent_down - nonevent_up
    return ReclassificationResult(
        event_improvement=float(event_improvement),
        nonevent_improvement=float(nonevent_improvement),
        net_reclassification=float(event_improvement + nonevent_improvement),
    )


def categorical_nri(
    outcome: ArrayLike,
    reference_risk: ArrayLike,
    candidate_risk: ArrayLike,
    boundaries: ArrayLike,
) -> ReclassificationResult:
    outcomes = np.asarray(outcome, dtype=np.int64)
    reference = np.digitize(np.asarray(reference_risk, dtype=np.float64), boundaries)
    candidate = np.digitize(np.asarray(candidate_risk, dtype=np.float64), boundaries)
    return category_free_nri(outcomes, reference, candidate)


def integrated_discrimination_improvement(
    outcome: ArrayLike,
    reference_risk: ArrayLike,
    candidate_risk: ArrayLike,
) -> float:
    outcomes = np.asarray(outcome, dtype=np.int64)
    reference = np.asarray(reference_risk, dtype=np.float64)
    candidate = np.asarray(candidate_risk, dtype=np.float64)
    events = outcomes == 1
    nonevents = outcomes == 0
    reference_slope = np.mean(reference[events]) - np.mean(reference[nonevents])
    candidate_slope = np.mean(candidate[events]) - np.mean(candidate[nonevents])
    return float(candidate_slope - reference_slope)


def risk_categories(risk: ArrayLike, boundaries: ArrayLike) -> np.ndarray:
    values = np.asarray(risk, dtype=np.float64)
    edges = np.asarray(boundaries, dtype=np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("risk boundaries")
    return np.digitize(values, edges)

