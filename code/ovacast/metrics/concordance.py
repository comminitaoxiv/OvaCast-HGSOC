from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ConcordanceResult:
    concordant: float
    discordant: float
    tied_risk: float
    comparable: float
    estimate: float


def _arrays(
    time: ArrayLike,
    event: ArrayLike,
    risk: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    risks = np.asarray(risk, dtype=np.float64)
    if times.ndim != 1 or events.ndim != 1 or risks.ndim != 1:
        raise ValueError("metric dimensions")
    if not (len(times) == len(events) == len(risks)):
        raise ValueError("metric lengths")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(risks)):
        raise ValueError("finite metric inputs")
    return times, events, risks


def concordance_details(
    time: ArrayLike,
    event: ArrayLike,
    risk: ArrayLike,
) -> ConcordanceResult:
    times, events, risks = _arrays(time, event, risk)
    concordant = 0.0
    discordant = 0.0
    tied = 0.0
    for left in range(len(times)):
        for right in range(left + 1, len(times)):
            if times[left] == times[right]:
                continue
            earlier = left if times[left] < times[right] else right
            later = right if earlier == left else left
            if events[earlier] == 0:
                continue
            difference = risks[earlier] - risks[later]
            if difference > 0:
                concordant += 1
            elif difference < 0:
                discordant += 1
            else:
                tied += 1
    comparable = concordant + discordant + tied
    estimate = (concordant + 0.5 * tied) / comparable if comparable else float("nan")
    return ConcordanceResult(concordant, discordant, tied, comparable, estimate)


def concordance_index(time: ArrayLike, event: ArrayLike, risk: ArrayLike) -> float:
    return concordance_details(time, event, risk).estimate


def comparable_pairs(time: ArrayLike, event: ArrayLike) -> NDArray[np.bool_]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event, dtype=np.int64)
    earlier = times[:, None] < times[None, :]
    return earlier & events[:, None].astype(bool)


def per_patient_concordance(
    time: ArrayLike,
    event: ArrayLike,
    risk: ArrayLike,
) -> NDArray[np.float64]:
    times, events, risks = _arrays(time, event, risk)
    values = np.full(len(times), np.nan, dtype=np.float64)
    for index in range(len(times)):
        successes = 0.0
        comparisons = 0.0
        for other in range(len(times)):
            if index == other or times[index] == times[other]:
                continue
            if times[index] < times[other] and events[index] == 1:
                expected = risks[index] > risks[other]
            elif times[other] < times[index] and events[other] == 1:
                expected = risks[other] > risks[index]
            else:
                continue
            comparisons += 1
            successes += float(expected)
            if risks[index] == risks[other]:
                successes += 0.5
        if comparisons:
            values[index] = successes / comparisons
    return values


def pooled_concordance(results: Iterable[ConcordanceResult]) -> ConcordanceResult:
    items = list(results)
    concordant = sum(item.concordant for item in items)
    discordant = sum(item.discordant for item in items)
    tied = sum(item.tied_risk for item in items)
    comparable = concordant + discordant + tied
    estimate = (concordant + 0.5 * tied) / comparable if comparable else float("nan")
    return ConcordanceResult(concordant, discordant, tied, comparable, estimate)

