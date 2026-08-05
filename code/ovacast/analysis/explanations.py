import re

import numpy as np
from scipy.stats import spearmanr


DRIVERS = frozenset(
    {
        "TP53",
        "BRCA1",
        "BRCA2",
        "RAD51",
        "PALB2",
        "ATM",
        "CCNE1",
        "RB1",
        "PTEN",
        "PIK3CA",
        "MYC",
        "NF1",
        "CDK12",
        "FANCD2",
        "FOXM1",
        "VIM",
        "SNAI2",
        "CDH1",
        "KRAS",
        "ERBB2",
        "AKT1",
        "BRAF",
        "CSMD3",
        "FAT3",
        "GABRA6",
    }
)


def entities(text: str, pathway_ids: frozenset[str] = frozenset()) -> tuple[str, ...]:
    words = re.findall(r"\b[A-Z][A-Z0-9-]{1,15}\b", text)
    accepted = [word for word in words if word in DRIVERS or word in pathway_ids]
    return tuple(dict.fromkeys(accepted))


def driver_precision(explanations: list[str], limit: int = 10) -> float:
    extracted = [entity for text in explanations for entity in entities(text)][:limit]
    return sum(entity in DRIVERS for entity in extracted) / len(extracted) if extracted else 0.0


def driver_recall(explanations: list[str], relevant: frozenset[str] = DRIVERS) -> float:
    recovered = {entity for text in explanations for entity in entities(text)}
    return len(recovered.intersection(relevant)) / len(relevant) if relevant else 0.0


def faithfulness(importances: np.ndarray, sensitivities: np.ndarray) -> float:
    correlation = spearmanr(importances, sensitivities).statistic
    return float(correlation)


def sufficiency(full_logits: np.ndarray, selected_logits: np.ndarray) -> float:
    residual = np.sum((full_logits - selected_logits) ** 2)
    total = np.sum((full_logits - np.mean(full_logits)) ** 2)
    return float(1 - residual / total) if total else 0.0


def cohens_kappa(first: np.ndarray, second: np.ndarray, categories: int = 5) -> float:
    agreement = float(np.mean(first == second))
    expected = 0.0
    for category in range(1, categories + 1):
        expected += np.mean(first == category) * np.mean(second == category)
    return (agreement - expected) / (1 - expected) if expected < 1 else 1.0
