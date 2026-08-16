"""Named categorical vocabularies for fixed-topology chemical diffusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpcr_ringgen.utils.io import load_yaml_config


def _normalize_labels(name: str, values: object) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    labels = tuple(values)
    if len(labels) < 2:
        raise ValueError(f"{name} must contain at least two labels")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"every {name} label must be a non-empty string")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} labels must be unique")
    return labels


@dataclass(frozen=True)
class ChemicalVocabulary:
    """Stable index-to-name mapping for generated atom and bond categories."""

    atom_types: tuple[str, ...]
    bond_types: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_types",
            _normalize_labels("atom_types", self.atom_types),
        )
        object.__setattr__(
            self,
            "bond_types",
            _normalize_labels("bond_types", self.bond_types),
        )

    @property
    def num_atom_types(self) -> int:
        return len(self.atom_types)

    @property
    def num_bond_types(self) -> int:
        return len(self.bond_types)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ChemicalVocabulary":
        section = config.get("chemical_vocabulary", config)
        if not isinstance(section, Mapping):
            raise TypeError("chemical_vocabulary config must be a mapping")
        missing = [name for name in ("atom_types", "bond_types") if name not in section]
        if missing:
            raise KeyError(
                "missing chemical vocabulary keys: " + ", ".join(missing)
            )
        return cls(
            atom_types=section["atom_types"],
            bond_types=section["bond_types"],
        )


def load_chemical_vocabulary(path: str | Path) -> ChemicalVocabulary:
    """Load atom and existing-edge bond labels from YAML."""

    return ChemicalVocabulary.from_mapping(load_yaml_config(path))


__all__ = ["ChemicalVocabulary", "load_chemical_vocabulary"]
