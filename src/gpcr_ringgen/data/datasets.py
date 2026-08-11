"""Dataset interfaces; no molecular data schema is selected yet."""

from abc import ABC, abstractmethod
from typing import Any


class MolecularDataset(ABC):
    """Minimal dataset boundary for future implementations."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of records."""

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return one record without prescribing its final schema."""
