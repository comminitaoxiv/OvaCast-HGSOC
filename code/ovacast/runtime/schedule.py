from math import cos, pi

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer: Optimizer, total_steps: int, warmup_ratio: float = 0.1, final_ratio: float = 0.1
) -> LambdaLR:
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + cos(pi * min(progress, 1.0)))

    return LambdaLR(optimizer, scale)
