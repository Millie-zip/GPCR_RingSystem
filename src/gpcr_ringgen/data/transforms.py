"""Transform boundary with no atom or graph schema assumption."""

from abc import ABC, abstractmethod
from typing import Any


class MolecularTransform(ABC):
    """Transform one future molecular record."""

    @abstractmethod
    def __call__(self, record: Any) -> Any:
        """Return a transformed record."""
