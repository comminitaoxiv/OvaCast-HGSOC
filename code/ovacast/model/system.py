from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ovacast.records.types import ModelResult


@dataclass(frozen=True)
class OvaCastConfig:
    backbone: str = "BioMistral/BioMistral-7B"
    hidden_size: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class SurvivalHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden).squeeze(-1)


class OvaCast(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int = 4096) -> None:
        super().__init__()
        self.backbone = backbone
        self.survival_head = SurvivalHead(hidden_size)

    @classmethod
    def load(cls, config: OvaCastConfig, trainable: bool = True) -> "OvaCast":
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            config.backbone, torch_dtype=torch.bfloat16, use_cache=False
        )
        adapter = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(base, adapter) if trainable else base
        return cls(backbone, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelResult:
        outputs: Any = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        lengths = attention_mask.sum(dim=1).long().sub(1).clamp_min(0)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        hidden = outputs.hidden_states[-1][rows, lengths]
        risk = self.survival_head(hidden.float())
        return ModelResult(risk, outputs.logits if labels is not None else None, hidden)
