"""Reproducibility helper for the dependencies used by the skeleton."""

import random

import numpy as np
import torch


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without assuming CUDA exists."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """Create a seeded local generator without mutating global RNG state."""

    _validate_seed(seed)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


__all__ = ["make_torch_generator", "set_seed"]
