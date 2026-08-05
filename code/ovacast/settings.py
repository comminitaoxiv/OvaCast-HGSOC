from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    context_length: int
    hidden_size: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class PhaseConfig:
    genomic_pretraining: int
    multimodal_finetuning: int
    explanation_tuning: int


@dataclass(frozen=True)
class TrainingConfig:
    phases: PhaseConfig
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_clip_norm: float
    precision: str
    modality_dropout: float
    explanation_loss_weight: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class DataConfig:
    pathways: int
    minimum_pathway_genes: int
    top_genes: int
    train_fraction: float
    validation_fraction_within_train: float


@dataclass(frozen=True)
class ComputeConfig:
    accelerator: str
    devices: int
    cuda: str
    expected_gpu_hours: float


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    compute: ComputeConfig


def _mapping(value: Any, key: str) -> dict[str, Any]:
    item = value[key]
    if not isinstance(item, dict):
        raise TypeError(key)
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value[key]
    if not isinstance(item, int):
        raise TypeError(key)
    return item


def _number(value: dict[str, Any], key: str) -> float:
    item = value[key]
    if not isinstance(item, (int, float)):
        raise TypeError(key)
    return float(item)


def _string(value: dict[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise TypeError(key)
    return item


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration")
    model = _mapping(raw, "model")
    training = _mapping(raw, "training")
    phases = _mapping(training, "phases")
    data = _mapping(raw, "data")
    compute = _mapping(raw, "compute")
    target_modules = model["target_modules"]
    seeds = training["seeds"]
    if not isinstance(target_modules, list) or not all(isinstance(x, str) for x in target_modules):
        raise TypeError("target_modules")
    if not isinstance(seeds, list) or not all(isinstance(x, int) for x in seeds):
        raise TypeError("seeds")
    result = ExperimentConfig(
        model=ModelConfig(
            backbone=_string(model, "backbone"),
            context_length=_integer(model, "context_length"),
            hidden_size=_integer(model, "hidden_size"),
            lora_rank=_integer(model, "lora_rank"),
            lora_alpha=_integer(model, "lora_alpha"),
            lora_dropout=_number(model, "lora_dropout"),
            target_modules=tuple(target_modules),
        ),
        training=TrainingConfig(
            phases=PhaseConfig(
                genomic_pretraining=_integer(phases, "genomic_pretraining"),
                multimodal_finetuning=_integer(phases, "multimodal_finetuning"),
                explanation_tuning=_integer(phases, "explanation_tuning"),
            ),
            batch_size=_integer(training, "batch_size"),
            gradient_accumulation_steps=_integer(training, "gradient_accumulation_steps"),
            effective_batch_size=_integer(training, "effective_batch_size"),
            learning_rate=_number(training, "learning_rate"),
            weight_decay=_number(training, "weight_decay"),
            warmup_ratio=_number(training, "warmup_ratio"),
            gradient_clip_norm=_number(training, "gradient_clip_norm"),
            precision=_string(training, "precision"),
            modality_dropout=_number(training, "modality_dropout"),
            explanation_loss_weight=_number(training, "explanation_loss_weight"),
            seeds=tuple(seeds),
        ),
        data=DataConfig(
            pathways=_integer(data, "pathways"),
            minimum_pathway_genes=_integer(data, "minimum_pathway_genes"),
            top_genes=_integer(data, "top_genes"),
            train_fraction=_number(data, "train_fraction"),
            validation_fraction_within_train=_number(data, "validation_fraction_within_train"),
        ),
        compute=ComputeConfig(
            accelerator=_string(compute, "accelerator"),
            devices=_integer(compute, "devices"),
            cuda=_string(compute, "cuda"),
            expected_gpu_hours=_number(compute, "expected_gpu_hours"),
        ),
    )
    expected = result.training.batch_size * result.training.gradient_accumulation_steps
    if expected != result.training.effective_batch_size:
        raise ValueError("effective_batch_size")
    return result
