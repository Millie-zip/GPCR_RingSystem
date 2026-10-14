"""Frozen categorical vocabularies shared by data adapters and models."""

from __future__ import annotations

from gpcr_ringgen.vocabulary import (
    ATOM_ID_TO_TYPE,
    ATOM_TYPE_COUNT,
    ATOM_TYPE_TO_ID,
    BOND_ID_TO_TYPE,
    BOND_TYPE_COUNT,
    BOND_TYPE_TO_ID,
)

__all__ = [
    "ATOM_ID_TO_TYPE",
    "ATOM_TYPE_COUNT",
    "ATOM_TYPE_TO_ID",
    "BOND_ID_TO_TYPE",
    "BOND_TYPE_COUNT",
    "BOND_TYPE_TO_ID",
]
