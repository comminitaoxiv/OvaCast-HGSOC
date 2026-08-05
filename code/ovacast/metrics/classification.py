from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float
    average_precision: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float


@dataclass(frozen=True)
class MulticlassMetrics:
    macro_auroc: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    accuracy: float
    per_class_auroc: NDArray[np.float64]
    per_class_precision: NDArray[np.float64]
    per_class_recall: NDArray[np.float64]
    per_class_f1: NDArray[np.float64]


def sigmoid(values: ArrayLike) -> NDArray[np.float64]:
    logits = np.asarray(values, dtype=np.float64)
    positive = logits >= 0
    output = np.empty_like(logits)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def softmax(values: ArrayLike) -> NDArray[np.float64]:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def binary_metrics(
    labels: ArrayLike,
    scores: ArrayLike,
    threshold: float = 0.5,
) -> BinaryMetrics:
    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)
    return BinaryMetrics(
        auroc=float(roc_auc_score(targets, probabilities)),
        average_precision=float(average_precision_score(targets, probabilities)),
        accuracy=float(accuracy_score(targets, predictions)),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        recall=float(recall_score(targets, predictions, zero_division=0)),
        f1=float(f1_score(targets, predictions, zero_division=0)),
        specificity=float(specificity),
    )


def multiclass_metrics(labels: ArrayLike, probabilities: ArrayLike) -> MulticlassMetrics:
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(scores, axis=1)
    classes = np.arange(scores.shape[1])
    one_hot = np.eye(scores.shape[1], dtype=np.int64)[targets]
    per_auroc = np.asarray(
        [
            roc_auc_score(one_hot[:, class_index], scores[:, class_index])
            for class_index in classes
        ],
        dtype=np.float64,
    )
    precision = precision_score(targets, predictions, labels=classes, average=None, zero_division=0)
    recall = recall_score(targets, predictions, labels=classes, average=None, zero_division=0)
    f1 = f1_score(targets, predictions, labels=classes, average=None, zero_division=0)
    return MulticlassMetrics(
        macro_auroc=float(np.mean(per_auroc)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        accuracy=float(accuracy_score(targets, predictions)),
        per_class_auroc=per_auroc,
        per_class_precision=np.asarray(precision, dtype=np.float64),
        per_class_recall=np.asarray(recall, dtype=np.float64),
        per_class_f1=np.asarray(f1, dtype=np.float64),
    )


def optimal_youden_threshold(labels: ArrayLike, scores: ArrayLike) -> float:
    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(probabilities)
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in candidates:
        predictions = probabilities >= threshold
        sensitivity = np.mean(predictions[targets == 1])
        specificity = np.mean(~predictions[targets == 0])
        value = sensitivity + specificity - 1
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold

