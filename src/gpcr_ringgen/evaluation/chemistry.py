"""Chemistry evaluation placeholder."""

from typing import Any, Sequence

from gpcr_ringgen.core.types import GeneratedMolecule


def evaluate_chemistry(molecules: Sequence[GeneratedMolecule]) -> dict[str, Any]:
    """Evaluate chemistry after a validity toolkit is selected."""
    raise NotImplementedError("Chemistry evaluation is not implemented.")
