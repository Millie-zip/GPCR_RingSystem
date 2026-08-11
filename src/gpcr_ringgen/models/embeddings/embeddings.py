"""Embedding boundary without atom, bond, time, or ring vocabularies."""

from abc import ABC, abstractmethod
from typing import Any


class FeatureEmbedding(ABC):
    """Future feature embedding contract."""

    @abstractmethod
    def encode(self, values: Any) -> Any:
        """Encode values after the representation is selected."""
