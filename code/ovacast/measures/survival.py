from math import exp

import numpy as np


def concordance_index(times: np.ndarray, risk: np.ndarray, events: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            early, late = (i, j) if times[i] < times[j] else (j, i)
            if events[early] != 1:
                continue
            comparable += 1.0
            if risk[early] == risk[late]:
                concordant += 0.5
            elif risk[early] > risk[late]:
                concordant += 1.0
    return concordant / comparable if comparable else float("nan")


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(times[events == 1])
    survival = []
    estimate = 1.0
    for time in unique:
        at_risk = np.sum(times >= time)
        deaths = np.sum((times == time) & (events == 1))
        estimate *= 1.0 - deaths / at_risk
        survival.append(estimate)
    return unique, np.asarray(survival)


def breslow_baseline(
    times: np.ndarray, risk: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(times[events == 1])
    cumulative = []
    current = 0.0
    hazards = np.exp(risk)
    for time in unique:
        deaths = np.sum((times == time) & (events == 1))
        denominator = np.sum(hazards[times >= time])
        current += deaths / denominator
        cumulative.append(current)
    return unique, np.asarray(cumulative)


def survival_probability(risk: np.ndarray, baseline_hazard: float) -> np.ndarray:
    return np.exp(-baseline_hazard * np.exp(risk))


def integrated_brier_score(
    times: np.ndarray, events: np.ndarray, predictions: np.ndarray, grid: np.ndarray
) -> float:
    scores = []
    for index, landmark in enumerate(grid):
        observed = (times > landmark).astype(float)
        eligible = (times > landmark) | (events == 1)
        if np.any(eligible):
            scores.append(np.mean((observed[eligible] - predictions[eligible, index]) ** 2))
    return (
        float(np.trapz(scores, grid[: len(scores)]) / (grid[len(scores) - 1] - grid[0]))
        if len(scores) > 1
        else float("nan")
    )


def logrank_statistic(
    times: np.ndarray, events: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for time in np.unique(times[events == 1]):
        at_risk = times >= time
        deaths = (times == time) & (events == 1)
        n = np.sum(at_risk)
        n1 = np.sum(at_risk & groups)
        d = np.sum(deaths)
        d1 = np.sum(deaths & groups)
        if n <= 1:
            continue
        observed += d1
        expected += d * n1 / n
        variance += n1 * (n - n1) * d * (n - d) / (n * n * (n - 1))
    statistic = (observed - expected) ** 2 / variance if variance > 0 else 0.0
    return statistic, exp(-0.5 * statistic)
