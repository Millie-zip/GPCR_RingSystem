"""Frozen categorical vocabularies shared by data adapters and models."""

from __future__ import annotations

from types import MappingProxyType

ATOM_TYPE_TO_ID = MappingProxyType(
    {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "P": 4,
        "F": 5,
        "Cl": 6,
        "Br": 7,
        "I": 8,
    }
)
BOND_TYPE_TO_ID = MappingProxyType(
    {
        "SINGLE": 0,
        "DOUBLE": 1,
        "TRIPLE": 2,
        "AROMATIC": 3,
    }
)

ATOM_TYPE_COUNT = len(ATOM_TYPE_TO_ID)
BOND_TYPE_COUNT = len(BOND_TYPE_TO_ID)

ATOM_ID_TO_TYPE = tuple(ATOM_TYPE_TO_ID)
BOND_ID_TO_TYPE = tuple(BOND_TYPE_TO_ID)

__all__ = [
    "ATOM_ID_TO_TYPE",
    "ATOM_TYPE_COUNT",
    "ATOM_TYPE_TO_ID",
    "BOND_ID_TO_TYPE",
    "BOND_TYPE_COUNT",
    "BOND_TYPE_TO_ID",
]
