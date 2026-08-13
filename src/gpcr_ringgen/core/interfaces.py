"""Abstract collaboration boundaries for future scientific modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from gpcr_ringgen.core.types import (
    DiffusionState,
    GeneratedMolecule,
    PocketBatch,
    PocketField,
    RingTopology,
    ScaffoldCondition,
)


class TopologyProvider(ABC):
    """Produce a topology without prescribing its scientific encoding."""

    @abstractmethod
    def provide(self) -> RingTopology:
        """Return one topology."""


class ScaffoldProvider(ABC):
    """Build a scaffold condition from a shared topology object."""

    @abstractmethod
    def provide(self, topology: RingTopology) -> ScaffoldCondition:
        """Return one scaffold condition."""


class PocketFieldProvider(ABC):
    """Construct an optional field without fixing its representation or algorithm."""

    @abstractmethod
    def build(self, pocket: PocketBatch) -> PocketField:
        """Return field data compatible with the stable PocketField protocol."""


class DiffusionModel(ABC):
    """Transform a state under a condition at a given timestep."""

    @abstractmethod
    def forward(
        self,
        state: DiffusionState,
        condition: ScaffoldCondition,
        timestep: int,
    ) -> DiffusionState:
        """Return a transformed state."""


class Constraint(ABC):
    """Optional state constraint boundary."""

    @abstractmethod
    def apply(self, state: DiffusionState) -> DiffusionState:
        """Return the constrained state."""


class Guidance(ABC):
    """Optional sampler guidance boundary."""

    @abstractmethod
    def apply(self, state: DiffusionState, timestep: int) -> DiffusionState:
        """Return a guided state."""


class Sampler(ABC):
    """Sampling is independent from concrete model implementations."""

    @abstractmethod
    def sample(
        self,
        model: DiffusionModel,
        condition: ScaffoldCondition,
        guidance: Guidance | None = None,
        num_samples: int = 1,
    ) -> Sequence[GeneratedMolecule]:
        """Return generated records; guidance must remain optional."""


class Evaluator(ABC):
    """Evaluate generated records without fixing a metric suite."""

    @abstractmethod
    def evaluate(self, molecules: Sequence[GeneratedMolecule]) -> dict[str, Any]:
        """Return named evaluation results."""
