"""Runtime helpers: deterministic seeds, file I/O, and metadata."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class RunMetadata:
    seed: int
    torch_version: str
    cuda_available: bool
    device_count: int
    note: str = ""


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_run_metadata(seed: int, note: str = "") -> RunMetadata:
    return RunMetadata(
        seed=seed,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        note=note,
    )


def save_run_metadata(path: str | Path, metadata: RunMetadata) -> None:
    save_json(path, asdict(metadata))
