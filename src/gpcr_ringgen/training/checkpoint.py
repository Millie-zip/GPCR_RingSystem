"""Minimal checkpoint path helper; serialization remains framework-owned."""

from pathlib import Path

from gpcr_ringgen.utils.io import ensure_directory


def checkpoint_path(directory: str | Path, name: str) -> Path:
    """Return a safe checkpoint path with a conventional extension."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("Checkpoint name must not be empty.")
    return ensure_directory(directory) / f"{normalized}.ckpt"
