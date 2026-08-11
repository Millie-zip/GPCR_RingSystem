"""Lightweight sampling callback boundary."""

from abc import ABC

from gpcr_ringgen.core.types import GeneratedMolecule


class SamplingCallback(ABC):
    """Observe produced records without controlling sampling internals."""

    def on_sample(self, molecule: GeneratedMolecule) -> None:
        """Handle one generated record."""
