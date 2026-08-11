"""Loss interface without a selected scientific objective."""

from abc import ABC, abstractmethod
from typing import Any


class Loss(ABC):
    """Future loss contract."""

    @abstractmethod
    def compute(self, prediction: Any, target: Any) -> Any:
        """Compute a future research loss."""
