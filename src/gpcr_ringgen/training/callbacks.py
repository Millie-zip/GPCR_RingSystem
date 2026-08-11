"""Lightweight training callback boundary."""

from abc import ABC


class TrainingCallback(ABC):
    """Observe future training lifecycle events."""

    def on_epoch_end(self, epoch: int) -> None:
        """Handle the end of one future training epoch."""
