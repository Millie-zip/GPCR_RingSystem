"""Frozen chemical labels used by the real-data converter."""

from __future__ import annotations

from gpcr_ringgen.core.vocab import (
    ATOM_ID_TO_TYPE as ATOM_TYPES,
    ATOM_TYPE_TO_ID,
    BOND_ID_TO_TYPE as BOND_TYPES,
    BOND_TYPE_TO_ID,
)


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
