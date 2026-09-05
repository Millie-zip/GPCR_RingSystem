"""Pluggable RAscore teacher and cached label generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpcr_ringgen.training.ra_reward_data import (
    build_rascore_label_cache,
    load_score_callable,
)


class RAScoreTeacher:
    """Small adapter around an installed RAscore implementation."""

    def __init__(self, scorer: Callable[[str], float], *, version: str) -> None:
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer
        self.version = version

    def score_smiles(self, smiles: str) -> float:
        return float(self.scorer(smiles))

    def build_cache(
        self,
        records_path: str,
        output_path: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return build_rascore_label_cache(
            records_path,
            output_path,
            self.score_smiles,
            overwrite=overwrite,
            teacher_version=self.version,
        )


def load_rascore_teacher(import_path: str, *, version: str = "external_rascore") -> RAScoreTeacher:
    """Load a teacher function as ``module:function`` without hard-coding its package API."""
    return RAScoreTeacher(load_score_callable(import_path), version=version)


__all__ = ["RAScoreTeacher", "load_rascore_teacher"]
