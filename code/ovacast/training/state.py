from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainingState:
    phase: str
    epoch: int
    global_step: int
    best_validation_loss: float
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    state: TrainingState,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": asdict(state),
        "rng_state": rng_state(),
    }


def atomic_save(payload: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    map_location: str | torch.device = "cpu",
) -> TrainingState:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng_state"])
    raw = payload["training_state"]
    return TrainingState(
        phase=str(raw["phase"]),
        epoch=int(raw["epoch"]),
        global_step=int(raw["global_step"]),
        best_validation_loss=float(raw["best_validation_loss"]),
        seed=int(raw["seed"]),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_manifest(
    path: str | Path,
    state: TrainingState,
    configuration: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    payload = {
        "state": asdict(state),
        "configuration": configuration,
        "artifacts": artifacts,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".manifest.")
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
    os.replace(temporary, destination)

