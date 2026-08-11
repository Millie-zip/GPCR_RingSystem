"""Training interface without a diffusion training loop."""

from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """Future training orchestration boundary."""

    @abstractmethod
    def fit(self, config: dict[str, Any]) -> None:
        """Train after model, loss, and dataset decisions are approved."""


def run_training(config: dict[str, Any]) -> None:
    """Placeholder invoked by the training entrypoint."""
    raise NotImplementedError("The scientific training loop is not implemented.")
