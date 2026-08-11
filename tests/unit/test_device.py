"""Tests for portable device selection."""

import torch

from gpcr_ringgen.utils.device import resolve_device


def test_resolve_device_auto_returns_torch_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
