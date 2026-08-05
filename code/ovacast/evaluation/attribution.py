from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenAttribution:
    position: int
    token_id: int
    score: float
    modality: str


@dataclass(frozen=True)
class ModalityAttribution:
    modality: str
    signed_score: float
    absolute_score: float
    fraction: float


def gradient_input_attribution(
    scalar_output: Tensor,
    embeddings: Tensor,
) -> Tensor:
    gradient = torch.autograd.grad(
        scalar_output,
        embeddings,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return torch.sum(gradient * embeddings, dim=-1)


def integrated_gradients(
    forward: Callable[[Tensor], Tensor],
    embeddings: Tensor,
    baseline: Tensor | None = None,
    steps: int = 32,
) -> Tensor:
    if steps < 2:
        raise ValueError("steps")
    reference = torch.zeros_like(embeddings) if baseline is None else baseline
    if reference.shape != embeddings.shape:
        raise ValueError("baseline")
    difference = embeddings - reference
    accumulated = torch.zeros_like(embeddings)
    for alpha in torch.linspace(0.0, 1.0, steps, device=embeddings.device):
        interpolated = (reference + alpha * difference).detach().requires_grad_(True)
        output = forward(interpolated)
        gradient = torch.autograd.grad(output.sum(), interpolated)[0]
        accumulated = accumulated + gradient
    average = accumulated / steps
    return torch.sum(difference * average, dim=-1)


def normalize_attribution(values: Tensor) -> Tensor:
    denominator = values.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return values / denominator


def aggregate_spans(
    attribution: NDArray[np.float64],
    spans: Mapping[str, tuple[int, int]],
) -> tuple[ModalityAttribution, ...]:
    total = float(np.sum(np.abs(attribution)))
    results: list[ModalityAttribution] = []
    for modality, (start, end) in spans.items():
        selected = attribution[start:end]
        absolute = float(np.sum(np.abs(selected)))
        results.append(
            ModalityAttribution(
                modality=modality,
                signed_score=float(np.sum(selected)),
                absolute_score=absolute,
                fraction=absolute / total if total else 0.0,
            )
        )
    return tuple(results)


def rank_tokens(
    token_ids: Sequence[int],
    attribution: Sequence[float],
    modality_by_position: Sequence[str],
) -> tuple[TokenAttribution, ...]:
    if not (
        len(token_ids) == len(attribution) == len(modality_by_position)
    ):
        raise ValueError("token attribution dimensions")
    items = [
        TokenAttribution(
            position=index,
            token_id=int(token),
            score=float(score),
            modality=modality,
        )
        for index, (token, score, modality) in enumerate(
            zip(token_ids, attribution, modality_by_position, strict=True)
        )
    ]
    return tuple(sorted(items, key=lambda item: -abs(item.score)))


def pathway_attribution(
    token_attribution: Sequence[TokenAttribution],
    pathway_positions: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    position_scores = {item.position: item.score for item in token_attribution}
    return {
        pathway: float(sum(position_scores.get(position, 0.0) for position in positions))
        for pathway, positions in pathway_positions.items()
    }


def occlusion_difference(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    spans: Mapping[str, tuple[int, int]],
    replacement_token_id: int,
) -> dict[str, Tensor]:
    model.eval()
    with torch.no_grad():
        reference = model(input_ids=input_ids, attention_mask=attention_mask)
        reference_risk = reference.heads.log_hazard
        results: dict[str, Tensor] = {}
        for modality, (start, end) in spans.items():
            occluded = input_ids.clone()
            occluded[:, start:end] = replacement_token_id
            output = model(input_ids=occluded, attention_mask=attention_mask)
            results[modality] = reference_risk - output.heads.log_hazard
    return results


def top_pathways(
    scores: Mapping[str, float],
    count: int = 10,
) -> tuple[tuple[str, float], ...]:
    if count <= 0:
        raise ValueError("count")
    ordered = sorted(scores.items(), key=lambda item: (-abs(item[1]), item[0]))
    return tuple((name, float(value)) for name, value in ordered[:count])


def attribution_correlation(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    shared = sorted(set(first).intersection(second))
    if len(shared) < 2:
        return float("nan")
    left = np.asarray([first[name] for name in shared], dtype=np.float64)
    right = np.asarray([second[name] for name in shared], dtype=np.float64)
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def modality_balance(values: Sequence[ModalityAttribution]) -> float:
    fractions = np.asarray([value.fraction for value in values], dtype=np.float64)
    positive = fractions[fractions > 0]
    if positive.size < 2:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(entropy / np.log(len(positive)))
