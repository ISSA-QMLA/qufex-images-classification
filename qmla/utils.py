"""Runtime, reproducibility, and experiment metadata helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the CUDA/CPU build appropriate for this HPC system "
            "before running training or testing. It is intentionally excluded from project dependencies."
        ) from exc
    return torch


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> Any:
    torch = require_torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Load the cluster CUDA modules and install a matching "
            "PyTorch build, or set training.device='cpu'."
        )
    return device


def configure_accelerator(device: Any) -> None:
    torch = require_torch()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def package_versions() -> dict[str, str]:
    packages = [
        "qmla",
        "torch",
        "pennylane",
        "pennylane-lightning",
        "numpy",
        "pandas",
        "scikit-learn",
        "pillow",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    versions["python_hash_seed"] = os.environ.get("PYTHONHASHSEED", "not-set")
    return versions


def write_package_versions(directory: Path) -> Path:
    destination = directory / "package_versions.json"
    destination.write_text(json.dumps(package_versions(), indent=2), encoding="utf-8")
    return destination

