from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Iterator

from ovacast.records.types import PatientRecord


@dataclass(frozen=True)
class Partition:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def strata(records: list[PatientRecord]) -> dict[tuple[int, str], list[int]]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.event, record.clinical.stage)].append(index)
    return dict(groups)


def stratified_partition(
    records: list[PatientRecord],
    seed: int,
    test_fraction: float = 0.30,
    validation_fraction: float = 0.15,
) -> Partition:
    rng = Random(seed)
    train = []
    validation = []
    test = []
    for indices in strata(records).values():
        rng.shuffle(indices)
        test_count = round(len(indices) * test_fraction)
        remaining = indices[test_count:]
        validation_count = round(len(remaining) * validation_fraction)
        test.extend(indices[:test_count])
        validation.extend(remaining[:validation_count])
        train.extend(remaining[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return Partition(tuple(train), tuple(validation), tuple(test))


def repeated_stratified_folds(
    records: list[PatientRecord], folds: int = 5, repeats: int = 3, seed: int = 42
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    groups = strata(records)
    for repeat in range(repeats):
        buckets = [[] for _ in range(folds)]
        rng = Random(seed + repeat)
        for indices in groups.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            for offset, index in enumerate(shuffled):
                buckets[offset % folds].append(index)
        all_indices = set(range(len(records)))
        for bucket in buckets:
            validation = tuple(sorted(bucket))
            train = tuple(sorted(all_indices.difference(bucket)))
            yield train, validation
