from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit


@dataclass(frozen=True)
class SplitIndices:
    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]


def survival_strata(
    times: Sequence[float], events: Sequence[int], bins: int = 4
) -> NDArray[np.int64]:
    time_array = np.asarray(times, dtype=np.float64)
    event_array = np.asarray(events, dtype=np.int64)
    if len(time_array) != len(event_array):
        raise ValueError("outcome dimensions")
    quantiles = np.quantile(time_array, np.linspace(0, 1, bins + 1)[1:-1])
    time_bins = np.digitize(time_array, quantiles, right=True)
    return (event_array * bins + time_bins).astype(np.int64)


def primary_split(
    times: Sequence[float],
    events: Sequence[int],
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction_within_train: float = 0.15,
) -> SplitIndices:
    strata = survival_strata(times, events)
    indices = np.arange(len(strata), dtype=np.int64)
    outer = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    development, test = next(outer.split(indices, strata))
    inner_strata = strata[development]
    inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction_within_train,
        random_state=seed,
    )
    train_local, validation_local = next(inner.split(development, inner_strata))
    return SplitIndices(
        train=indices[development[train_local]],
        validation=indices[development[validation_local]],
        test=indices[test],
    )


def repeated_tier_one_folds(
    labels: Sequence[int],
    seed: int,
    folds: int = 5,
    repeats: int = 3,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    values = np.asarray(labels, dtype=np.int64)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    indices = np.arange(len(values), dtype=np.int64)
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(indices, values)
    ]


def validate_disjoint(split: SplitIndices) -> None:
    train = set(split.train.tolist())
    validation = set(split.validation.tolist())
    test = set(split.test.tolist())
    if train.intersection(validation) or train.intersection(test) or validation.intersection(test):
        raise ValueError("overlapping split")
    if len(train) + len(validation) + len(test) == 0:
        raise ValueError("empty split")


T = TypeVar("T")


def subset_sequence(values: Sequence[T], indices: NDArray[np.int64]) -> list[T]:
    return [values[int(index)] for index in indices]


def group_preserving_split(
    groups: Sequence[str],
    labels: Sequence[int],
    train_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if len(groups) != len(labels):
        raise ValueError("group dimensions")
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cutoff = round(len(unique) * train_fraction)
    train_groups = set(unique[:cutoff])
    train = np.asarray(
        [i for i, group in enumerate(groups) if group in train_groups], dtype=np.int64
    )
    test = np.asarray(
        [i for i, group in enumerate(groups) if group not in train_groups], dtype=np.int64
    )
    return train, test
