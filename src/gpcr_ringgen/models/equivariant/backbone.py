"""Equivariant backbone boundary without choosing EGNN, SE(3), or peers."""

from abc import ABC, abstractmethod
from typing import Any


class EquivariantBackbone(ABC):
    """Future equivariant backbone contract."""

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Return transformed features after an architecture is selected."""
