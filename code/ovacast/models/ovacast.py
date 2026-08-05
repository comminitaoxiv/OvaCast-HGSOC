from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

from ovacast.models.heads import ClinicalHeads, HeadOutput, last_valid_hidden


@dataclass(frozen=True)
class OvaCastOutput:
    heads: HeadOutput
    language_logits: Tensor
    hidden: Tensor


class OvaCast(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_size: int = 4096,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.backbone = get_peft_model(base, lora)
        self.heads = ClinicalHeads(hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> OvaCastOutput:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        final_hidden = output.hidden_states[-1]
        pooled = last_valid_hidden(final_hidden, attention_mask)
        return OvaCastOutput(
            heads=self.heads(pooled),
            language_logits=output.logits,
            hidden=pooled,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapter_state(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def merge_adapter(self) -> nn.Module:
        merged: Any = self.backbone.merge_and_unload()
        if not isinstance(merged, nn.Module):
            raise TypeError("merged adapter")
        return merged
