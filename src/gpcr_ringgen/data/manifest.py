"""Versioned real-data feature manifest and deterministic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gpcr_ringgen.data.vocabulary import (
    ATOM_TYPES, BOND_TYPES, normalize_bond_type, normalize_element,
)
from gpcr_ringgen.utils.io import load_yaml_config


DEFAULT_ATOM_FEATURES: tuple[str, ...] = (
    "element_C", "element_N", "element_O", "element_S", "element_P",
    "element_F", "element_Cl", "element_Br", "element_I",
    "formal_charge", "aromatic", "degree_scaled", "total_h_scaled",
    "ring_atom", "atomic_num_scaled", "chiral",
)
DEFAULT_EDGE_FEATURES: tuple[str, ...] = (
    "bond_SINGLE", "bond_DOUBLE", "bond_TRIPLE", "bond_AROMATIC",
    "aromatic", "conjugated", "ring_bond", "stereo",
)


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value != 0)
    return float(isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y"})


@dataclass(frozen=True)
class FeatureManifest:
    """Feature names and schema used by one training asset library."""

    schema_version: str
    atom_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    descriptor_features: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureManifest":
        if not isinstance(value, Mapping):
            raise TypeError("feature manifest must be a mapping")
        version = value.get("schema_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("feature manifest schema_version is required")

        def names(key: str) -> tuple[str, ...]:
            items = value.get(key)
            if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
                raise ValueError(f"feature manifest {key} must be a list of non-empty strings")
            result = tuple(item.strip() for item in items)
            if len(set(result)) != len(result):
                raise ValueError(f"feature manifest {key} contains duplicate names")
            return result

        atom = names("atom_features")
        edge = names("edge_features")
        descriptor_value = value.get("descriptor_features", [])
        if not isinstance(descriptor_value, list) or any(not isinstance(item, str) for item in descriptor_value):
            raise ValueError("feature manifest descriptor_features must be a string list")
        descriptor = tuple(item.strip() for item in descriptor_value)
        if not atom or not edge:
            raise ValueError("atom_features and edge_features must not be empty")
        return cls(version.strip(), atom, edge, descriptor)

    @classmethod
    def from_file(cls, path: str | Path) -> "FeatureManifest":
        return cls.from_mapping(load_yaml_config(path))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atom_features": list(self.atom_features),
            "edge_features": list(self.edge_features),
            "descriptor_features": list(self.descriptor_features),
        }

    @property
    def atom_feature_dim(self) -> int:
        return len(self.atom_features)

    @property
    def edge_feature_dim(self) -> int:
        return len(self.edge_features)

    @property
    def descriptor_dim(self) -> int:
        return len(self.descriptor_features)

    def atom_vector(self, facts: Mapping[str, Any], *, ring_atom: bool) -> list[float]:
        element = normalize_element(_first(facts, ("symbol", "element", "element_symbol", "atom_symbol")))
        if element not in ATOM_TYPES:
            raise ValueError(f"element outside frozen vocabulary: {element}")
        values = {
            **{f"element_{name}": float(name == element) for name in ATOM_TYPES},
            "formal_charge": _number(_first(facts, ("formal_charge", "charge"))) / 4.0,
            "aromatic": _boolean(_first(facts, ("aromatic", "is_aromatic"))),
            "degree_scaled": _number(_first(facts, ("degree", "total_degree"))) / 8.0,
            "total_h_scaled": _number(_first(facts, ("total_h", "total_hydrogens", "hydrogen_count", "num_h"))) / 4.0,
            "ring_atom": float(ring_atom),
            "atomic_num_scaled": _number(_first(facts, ("atomic_num", "atomic_number"))) / 100.0,
            "chiral": float(str(_first(facts, ("chiral_tag", "chirality", "stereo"), "")).strip().upper() not in {"", "NONE", "UNSPECIFIED", "0"}),
        }
        return self._select(values, self.atom_features, "atom")

    def edge_vector(self, facts: Mapping[str, Any], *, ring_bond: bool) -> list[float]:
        kind_value = _first(facts, ("bond_type", "type", "bond_kind"))
        if kind_value is None:
            kind_value = str(_number(_first(facts, ("bond_order", "order"))))
        kind = normalize_bond_type(kind_value)
        if kind not in BOND_TYPES:
            raise ValueError(f"bond type outside frozen vocabulary: {kind}")
        values = {
            **{f"bond_{name}": float(name == kind) for name in BOND_TYPES},
            "aromatic": _boolean(_first(facts, ("aromatic", "is_aromatic"))),
            "conjugated": _boolean(_first(facts, ("conjugated", "is_conjugated"))),
            "ring_bond": float(ring_bond),
            "stereo": float(str(_first(facts, ("stereo", "stereo_type", "stereochemistry"), "")).strip().upper() not in {"", "NONE", "UNSPECIFIED", "0"}),
        }
        return self._select(values, self.edge_features, "edge")

    @staticmethod
    def _select(values: Mapping[str, float], names: Sequence[str], kind: str) -> list[float]:
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"unsupported {kind} feature names: {missing}")
        return [float(values[name]) for name in names]


__all__ = ["DEFAULT_ATOM_FEATURES", "DEFAULT_EDGE_FEATURES", "FeatureManifest"]
