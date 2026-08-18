"""Versioned feature-schema loading for real Stage 3 assets.

The project does not assign scientific meaning to feature columns in code.  A
manifest produced alongside the converted data is therefore required before a
training run can start.  This prevents a tensor with the right width but the
wrong semantics from silently entering the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.utils.io import load_yaml_config


def _names(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"feature manifest {field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(f"feature manifest {field} contains an invalid name")
        result.append(item.strip())
    if not allow_empty and not result:
        raise ConfigurationError(f"feature manifest {field} must not be empty")
    if len(set(result)) != len(result):
        raise ConfigurationError(f"feature manifest {field} contains duplicate names")
    return tuple(result)


@dataclass(frozen=True)
class FeatureManifest:
    """The frozen column contract used by one converted asset library."""

    schema_version: str
    atom_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    descriptor_features: tuple[str, ...]
    pocket_features: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureManifest":
        if not isinstance(value, Mapping):
            raise ConfigurationError("feature manifest root must be a mapping")
        version = value.get("schema_version")
        if not isinstance(version, str) or not version.strip():
            raise ConfigurationError("feature manifest schema_version is required")
        features = value.get("features")
        if not isinstance(features, Mapping):
            raise ConfigurationError("feature manifest features must be a mapping")
        return cls(
            schema_version=version.strip(),
            atom_features=_names(features.get("atom"), "features.atom", allow_empty=False),
            edge_features=_names(features.get("edge"), "features.edge", allow_empty=False),
            descriptor_features=_names(
                features.get("descriptor", []),
                "features.descriptor",
                allow_empty=True,
            ),
            pocket_features=_names(
                features.get("pocket", []),
                "features.pocket",
                allow_empty=True,
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "FeatureManifest":
        return cls.from_mapping(load_yaml_config(path))

    @property
    def atom_feature_dim(self) -> int:
        return len(self.atom_features)

    @property
    def edge_feature_dim(self) -> int:
        return len(self.edge_features)

    @property
    def descriptor_dim(self) -> int:
        return len(self.descriptor_features)

    @property
    def pocket_feature_dim(self) -> int:
        return len(self.pocket_features)


__all__ = ["FeatureManifest"]
