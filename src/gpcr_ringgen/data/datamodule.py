"""Future orchestration boundary for datasets and data loaders."""

from abc import ABC, abstractmethod


class DataModule(ABC):
    """Prepare future datasets without fixing a framework-specific API."""

    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare data resources."""
