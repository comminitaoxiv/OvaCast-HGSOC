import numpy as np


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for label in np.unique(labels):
        mask = labels == label
        recalls.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(recalls))


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for expected, predicted in zip(labels, predictions, strict=True):
        matrix[int(expected), int(predicted)] += 1
    return matrix


def per_class_statistics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(
        true_positive,
        matrix.sum(axis=0),
        out=np.zeros_like(true_positive),
        where=matrix.sum(axis=0) != 0,
    )
    recall = np.divide(
        true_positive,
        matrix.sum(axis=1),
        out=np.zeros_like(true_positive),
        where=matrix.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def weighted_f1(labels: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    matrix = confusion_matrix(labels, predictions, classes)
    values = per_class_statistics(matrix)["f1"]
    weights = matrix.sum(axis=1)
    return float(np.average(values, weights=weights))
