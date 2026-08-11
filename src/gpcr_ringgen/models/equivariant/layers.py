"""Equivariant layer boundary without concrete tensor conventions."""

from abc import ABC, abstractmethod
from typing import Any


class EquivariantLayer(ABC):
    """Future layer contract."""

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Return transformed features."""
