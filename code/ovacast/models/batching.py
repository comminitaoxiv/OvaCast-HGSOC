from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ClinicalBatch:
    input_ids: Tensor
    attention_mask: Tensor
    language_labels: Tensor
    survival_time: Tensor
    event: Tensor
    subtype: Tensor
    platinum: Tensor


class ClinicalCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int = 32768,
        label_pad_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length
        self.label_pad_id = label_pad_id

    def __call__(self, examples: Sequence[dict[str, object]]) -> ClinicalBatch:
        sequences = [str(example["sequence"]) for example in examples]
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.maximum_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_id
        return ClinicalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            language_labels=labels,
            survival_time=torch.tensor(
                [float(example["survival_time"]) for example in examples],
                dtype=torch.float32,
            ),
            event=torch.tensor([int(example["event"]) for example in examples], dtype=torch.long),
            subtype=torch.tensor(
                [int(example.get("subtype", -1)) for example in examples],
                dtype=torch.long,
            ),
            platinum=torch.tensor(
                [int(example.get("platinum", -1)) for example in examples],
                dtype=torch.long,
            ),
        )


def move_batch(batch: ClinicalBatch, device: torch.device) -> ClinicalBatch:
    return ClinicalBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        language_labels=batch.language_labels.to(device),
        survival_time=batch.survival_time.to(device),
        event=batch.event.to(device),
        subtype=batch.subtype.to(device),
        platinum=batch.platinum.to(device),
    )


def batch_size(batch: ClinicalBatch) -> int:
    return int(batch.input_ids.shape[0])


def valid_subtype_mask(batch: ClinicalBatch) -> Tensor:
    return batch.subtype >= 0


def valid_platinum_mask(batch: ClinicalBatch) -> Tensor:
    return batch.platinum >= 0


def valid_survival_mask(batch: ClinicalBatch) -> Tensor:
    return torch.isfinite(batch.survival_time) & (batch.survival_time > 0)
