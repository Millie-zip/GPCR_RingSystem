"""Pocket preprocessing placeholder."""

from pathlib import Path

from gpcr_ringgen.core.types import PocketRepresentation


def preprocess_pocket(path: str | Path) -> PocketRepresentation:
    """Parse a future pocket input after PDB and extraction choices are made."""
    raise NotImplementedError("Pocket preprocessing is not implemented.")
