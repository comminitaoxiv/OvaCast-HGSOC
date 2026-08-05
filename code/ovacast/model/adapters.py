import torch
from torch import nn


class LowRankProjection(nn.Module):
    def __init__(
        self, features_in: int, features_out: int, rank: int = 16, alpha: float = 32.0
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, features_in))
        self.b = nn.Parameter(torch.zeros(features_out, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs @ self.a.transpose(0, 1) @ self.b.transpose(0, 1)) * self.scale


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def freeze_except_adapters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = "lora_" in name or "survival_head" in name
