"""Batch collation placeholder."""

from typing import Any, Sequence


def collate_molecules(records: Sequence[Any]) -> Any:
    """Collate records after the batching schema is scientifically decided."""
    raise NotImplementedError("The molecular batching schema is not defined yet.")
