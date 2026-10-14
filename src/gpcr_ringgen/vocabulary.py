"""Canonical persisted chemical IDs, with no tensor-library dependencies."""

from __future__ import annotations

from types import MappingProxyType


# Preserve the original nine IDs and append elements retained by real data.
ATOM_ID_TO_TYPE = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si")
BOND_ID_TO_TYPE = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
ATOM_TYPE_TO_ID = MappingProxyType({name: index for index, name in enumerate(ATOM_ID_TO_TYPE)})
BOND_TYPE_TO_ID = MappingProxyType({name: index for index, name in enumerate(BOND_ID_TO_TYPE)})
ATOM_TYPE_COUNT = len(ATOM_ID_TO_TYPE)
BOND_TYPE_COUNT = len(BOND_ID_TO_TYPE)

__all__ = [
    "ATOM_ID_TO_TYPE",
    "ATOM_TYPE_COUNT",
    "ATOM_TYPE_TO_ID",
    "BOND_ID_TO_TYPE",
    "BOND_TYPE_COUNT",
    "BOND_TYPE_TO_ID",
]
