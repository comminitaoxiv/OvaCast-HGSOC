from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ovacast.evaluation.bootstrap import BootstrapInterval, bootstrap_interval
from ovacast.evaluation.calibration import (
    baseline_survival,
    integrated_brier_score,
    predict_survival,
)
from ovacast.evaluation.explanations import ExplanationScore
from ovacast.evaluation.reclassification import ReclassificationResult, category_free_nri
from ovacast.evaluation.hypothesis import TestResult, benjamini_hochberg, holm_bonferroni
from ovacast.metrics.classification import BinaryMetrics, MulticlassMetrics, binary_metrics
from ovacast.metrics.concordance import concordance_index
from ovacast.metrics.survival_curves import median_risk_groups


@dataclass(frozen=True)
class SurvivalEndpoint:
    concordance: float
    concordance_interval: BootstrapInterval
    integrated_brier: float
    risk_groups: NDArray[np.int64]


@dataclass(frozen=True)
class ExternalEndpoint:
    cohort: str
    sample_count: int
    concordance: float
    lower: float
    upper: float


@dataclass(frozen=True)
class TierOneEndpoint:
    fold: int
    repeat: int
    concordance: float
    sample_count: int


@dataclass(frozen=True)
class BaselineComparison:
    method: str
    estimate: float
    difference: float
    raw_p_value: float
    adjusted_p_value: float
    primary: bool


@dataclass(frozen=True)
class AblationEndpoint:
    name: str
    concordance: float
    delta: float
    modalities: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationEndpoint:
    driver_recovery: float
    faithfulness: float
    clinical_plausibility: float
    coherence: float
    interrater_kappa: float


@dataclass(frozen=True)
class EvaluationBundle:
    survival: SurvivalEndpoint
    external: tuple[ExternalEndpoint, ...]
    tier_one: tuple[TierOneEndpoint, ...]
    comparisons: tuple[BaselineComparison, ...]
    ablations: tuple[AblationEndpoint, ...]
    platinum: BinaryMetrics | None
    subtype: MulticlassMetrics | None
    explanations: ExplanationEndpoint | None
    reclassification: ReclassificationResult | None


def evaluate_survival(
    time: NDArray[np.float64],
    event: NDArray[np.int64],
    log_hazard: NDArray[np.float64],
    horizons: NDArray[np.float64],
    repetitions: int = 2000,
    seed: int = 42,
) -> SurvivalEndpoint:
    interval = bootstrap_interval(
        time,
        event,
        log_hazard,
        concordance_index,
        repetitions=repetitions,
        seed=seed,
    )
    baseline_time, baseline_values = baseline_survival(time, event, log_hazard)
    curves = predict_survival(
        baseline_time,
        baseline_values,
        log_hazard,
        horizons,
    )
    brier = integrated_brier_score(time, event, curves, horizons)
    return SurvivalEndpoint(
        concordance=interval.estimate,
        concordance_interval=interval,
        integrated_brier=brier,
        risk_groups=median_risk_groups(log_hazard),
    )


def evaluate_external(
    cohort: str,
    time: NDArray[np.float64],
    event: NDArray[np.int64],
    log_hazard: NDArray[np.float64],
    seed: int,
) -> ExternalEndpoint:
    interval = bootstrap_interval(
        time,
        event,
        log_hazard,
        concordance_index,
        repetitions=2000,
        seed=seed,
    )
    return ExternalEndpoint(
        cohort=cohort,
        sample_count=len(time),
        concordance=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
    )


def pool_external(
    cohorts: Mapping[str, tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]],
    seed: int,
) -> tuple[ExternalEndpoint, ...]:
    return tuple(
        evaluate_external(name, values[0], values[1], values[2], seed)
        for name, values in sorted(cohorts.items())
    )


def summarize_tier_one(
    predictions: Sequence[
        tuple[int, int, NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]
    ],
) -> tuple[TierOneEndpoint, ...]:
    return tuple(
        TierOneEndpoint(
            fold=fold,
            repeat=repeat,
            concordance=concordance_index(time, event, risk),
            sample_count=len(time),
        )
        for fold, repeat, time, event, risk in predictions
    )


def aggregate_tier_one(endpoints: Sequence[TierOneEndpoint]) -> tuple[float, float]:
    values = np.asarray([endpoint.concordance for endpoint in endpoints], dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def adjust_comparisons(
    methods: Sequence[str],
    estimates: Sequence[float],
    full_estimate: float,
    tests: Sequence[TestResult],
    primary_methods: Sequence[str],
) -> tuple[BaselineComparison, ...]:
    if not (len(methods) == len(estimates) == len(tests)):
        raise ValueError("comparison dimensions")
    raw = np.asarray([test.p_value for test in tests], dtype=np.float64)
    family_adjusted = benjamini_hochberg(raw)
    primary_positions = [index for index, method in enumerate(methods) if method in primary_methods]
    primary_raw = raw[primary_positions]
    primary_adjusted = holm_bonferroni(primary_raw)
    final = family_adjusted.copy()
    for position, adjusted in zip(primary_positions, primary_adjusted, strict=True):
        final[position] = adjusted
    return tuple(
        BaselineComparison(
            method=method,
            estimate=float(estimate),
            difference=float(full_estimate - estimate),
            raw_p_value=float(test.p_value),
            adjusted_p_value=float(adjusted),
            primary=method in primary_methods,
        )
        for method, estimate, test, adjusted in zip(
            methods,
            estimates,
            tests,
            final,
            strict=True,
        )
    )


def ablation_table(
    full_estimate: float,
    values: Mapping[str, tuple[float, tuple[str, ...]]],
) -> tuple[AblationEndpoint, ...]:
    return tuple(
        AblationEndpoint(
            name=name,
            concordance=estimate,
            delta=estimate - full_estimate,
            modalities=modalities,
        )
        for name, (estimate, modalities) in values.items()
    )


def explanation_endpoint(
    scores: Sequence[ExplanationScore],
    clinical_ratings: Sequence[float],
    kappa: float,
) -> ExplanationEndpoint:
    recovery = np.asarray([score.driver_recovery for score in scores], dtype=np.float64)
    faithfulness = np.asarray([score.faithfulness for score in scores], dtype=np.float64)
    coherence = np.asarray([score.cross_modal_coherence for score in scores], dtype=np.float64)
    return ExplanationEndpoint(
        driver_recovery=float(np.nanmean(recovery)),
        faithfulness=float(np.nanmean(faithfulness)),
        clinical_plausibility=float(np.mean(clinical_ratings)),
        coherence=float(np.nanmean(coherence)),
        interrater_kappa=float(kappa),
    )


def platinum_endpoint(labels: NDArray[np.int64], logits: NDArray[np.float64]) -> BinaryMetrics:
    probability = 1.0 / (1.0 + np.exp(-logits))
    return binary_metrics(labels, probability)


def reclassification_endpoint(
    outcomes: NDArray[np.int64],
    baseline_risk: NDArray[np.float64],
    ovacast_risk: NDArray[np.float64],
) -> ReclassificationResult:
    return category_free_nri(outcomes, baseline_risk, ovacast_risk)


def seed_summary(values: Mapping[int, float]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def require_expected_cohorts(cohorts: Mapping[str, object]) -> None:
    expected = {"TCGA-OV", "GSE26712", "GSE9891", "PTRC-HGSOC"}
    missing = expected.difference(cohorts)
    if missing:
        raise ValueError(",".join(sorted(missing)))


def reportable_difference(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline)


def confidence_text(interval: BootstrapInterval, digits: int = 3) -> str:
    return (
        f"{interval.estimate:.{digits}f} "
        f"({interval.lower:.{digits}f}-{interval.upper:.{digits}f})"
    )


def evaluate_by_seed(
    seeds: Sequence[int],
    callback: Callable[[int], float],
) -> dict[int, float]:
    return {seed: float(callback(seed)) for seed in seeds}
