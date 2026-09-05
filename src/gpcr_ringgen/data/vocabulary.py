"""Frozen chemical labels used by the real-data converter."""

from __future__ import annotations

from typing import Mapping


# Keep the original nine IDs stable and append elements present in the
# production model-input database.  The source coordinates/topology retain
# these atoms, so dropping them during materialization would change the graph.
ATOM_TYPES: tuple[str, ...] = (
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si"
)
BOND_TYPES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
ATOM_TYPE_TO_ID: Mapping[str, int] = {name: index for index, name in enumerate(ATOM_TYPES)}
BOND_TYPE_TO_ID: Mapping[str, int] = {name: index for index, name in enumerate(BOND_TYPES)}


def normalize_element(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("atom element must be a non-empty string")
    text = value.strip()
    return text[0].upper() + text[1:].lower()


def normalize_bond_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("bond type must be a non-empty string")
    text = value.strip().upper().replace(" ", "_")
    aliases = {
        "1": "SINGLE", "1.0": "SINGLE", "SINGLE_BOND": "SINGLE",
        "2": "DOUBLE", "2.0": "DOUBLE", "DOUBLE_BOND": "DOUBLE",
        "3": "TRIPLE", "3.0": "TRIPLE", "TRIPLE_BOND": "TRIPLE",
        "AROMATIC_BOND": "AROMATIC", "1.5": "AROMATIC",
    }
    return aliases.get(text, text)


__all__ = [
    "ATOM_TYPES", "BOND_TYPES", "ATOM_TYPE_TO_ID", "BOND_TYPE_TO_ID",
    "normalize_element", "normalize_bond_type",
]
