"""Tests for reproducible random state initialization."""

import random

import numpy as np
import torch

from gpcr_ringgen.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second
