"""Structural and test-time checks for serialized Stage 3 outputs.

These checks deliberately do not claim chemical validity, docking quality, or
pharmacological usefulness.  They verify the fixed-topology and numerical
contracts that can be checked without a real reference database.  Ring/noise
weak constraints belong here, after sampling, rather than in the training
loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _finite_nested(value: object, name: str) -> None:
    values = _sequence(value, name)
    for row in values:
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            _finite_nested(row, name)
        elif not isinstance(row, (int, float)) or not math.isfinite(float(row)):
            raise ValueError(f"{name} contains a non-finite value")


def _bools(value: object, name: str) -> list[bool]:
    values = list(_sequence(value, name))
    if any(not isinstance(item, bool) for item in values):
        raise ValueError(f"{name} must contain booleans")
    return values


def _sample_checks(sample: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        atom_mask = _bools(sample.get("atom_mask"), "atom_mask")
        local = _sequence(sample.get("local_coordinates"), "local_coordinates")
        global_coordinates = _sequence(
            sample.get("global_coordinates"), "global_coordinates"
        )
        if len(local) != len(atom_mask) or len(global_coordinates) != len(atom_mask):
            raise ValueError("coordinates and atom_mask must have the same length")
        _finite_nested(local, "local_coordinates")
        _finite_nested(global_coordinates, "global_coordinates")
        topology = sample.get("topology")
        if not isinstance(topology, Mapping):
            raise ValueError("topology must be an object")
        edge_index = _sequence(topology.get("edge_index"), "topology.edge_index")
        if len(edge_index) != 2:
            raise ValueError("topology.edge_index must have shape [2, E]")
        edge_mask = _bools(topology.get("edge_mask"), "topology.edge_mask")
        if len(edge_index[0]) != len(edge_index[1]) or len(edge_mask) != len(edge_index[0]):
            raise ValueError("edge_index and edge_mask must have matching lengths")
        _finite_nested(topology.get("atom_features"), "topology.atom_features")
        _finite_nested(topology.get("edge_features"), "topology.edge_features")
        seen: set[tuple[int, int]] = set()
        for source, target, valid in zip(edge_index[0], edge_index[1], edge_mask, strict=True):
            if not valid:
                continue
            if (
                isinstance(source, bool)
                or isinstance(target, bool)
                or not isinstance(source, int)
                or not isinstance(target, int)
            ):
                raise ValueError("valid edge endpoints must be integers")
            if source < 0 or target < 0 or source >= len(atom_mask) or target >= len(atom_mask):
                raise ValueError("valid edge endpoint is out of range")
            if source == target:
                raise ValueError("valid edge cannot be a self-loop")
            if not atom_mask[source] or not atom_mask[target]:
                raise ValueError("valid edge cannot reference a padded atom")
            key = tuple(sorted((source, target)))
            if key in seen:
                raise ValueError("duplicate undirected topology edge")
            seen.add(key)
        labels = sample.get("chemical_labels")
        if labels is not None:
            if not isinstance(labels, Mapping):
                raise ValueError("chemical_labels must be an object")
            atom_ids = _sequence(labels.get("atom_type_ids"), "atom_type_ids")
            bond_ids = _sequence(labels.get("bond_type_ids"), "bond_type_ids")
            if len(atom_ids) != len(atom_mask) or len(bond_ids) != len(edge_mask):
                raise ValueError("chemical labels do not align with masks")
            if any((not isinstance(item, int) or item < 0) for item in atom_ids if item is not None):
                raise ValueError("atom type ids must be non-negative integers")
            if any((not isinstance(item, int) or item < 0) for item in bond_ids if item is not None):
                raise ValueError("bond type ids must be non-negative integers")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def validate_generated_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return machine-readable structural validation for one output JSON."""

    if not isinstance(payload, Mapping):
        raise ValueError("output root must be an object")
    samples = payload.get("samples")
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("output.samples must be an array")
    invalid: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            invalid.append({"index": index, "errors": ["sample must be an object"]})
            continue
        errors = _sample_checks(sample)
        if errors:
            invalid.append({"index": index, "sample_id": sample.get("sample_id"), "errors": errors})
    return {
        "valid": not invalid and len(samples) > 0,
        "sample_count": len(samples),
        "invalid_samples": invalid,
        "scientific_scope": "structural_and_numerical_checks_only",
        "weak_constraints": "test_time_only",
    }


__all__ = ["validate_generated_output"]
