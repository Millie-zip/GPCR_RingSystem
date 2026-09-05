"""RAscore label cache and JSONL dataset for the independent reward model."""

from __future__ import annotations

import importlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from gpcr_ringgen.data.datasets import JsonlMolecularDataset
from gpcr_ringgen.data.records import TrainingRecord
from gpcr_ringgen.core.exceptions import ConfigurationError


def _smiles(record: TrainingRecord) -> str:
    value = record.metadata.get("canonical_isomeric_smiles")
    if value is None:
        value = record.metadata.get("smiles", record.metadata.get("original_smiles"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record {record.sample_id} has no canonical SMILES metadata")
    return value.strip()


class RAScoreLabelStore:
    """Read-only SQLite label cache keyed by canonical isomeric SMILES."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"RA label cache not found: {self.path}")
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        return self._connection

    def get(self, canonical_smiles: str) -> float:
        row = self._connect().execute(
            "SELECT ra_score FROM rascore_labels WHERE canonical_smiles=?",
            (canonical_smiles,),
        ).fetchone()
        if row is None:
            raise KeyError(f"RA label missing for canonical SMILES: {canonical_smiles}")
        return float(row[0])

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()


class RAScoreLabeledDataset(Dataset[tuple[TrainingRecord, Tensor]]):
    """Pair existing Stage 3 records with cached teacher rewards."""

    def __init__(
        self,
        records_path: str | Path,
        labels_path: str | Path,
        *,
        split: str | None = None,
        train_fraction: float = 0.9,
        max_records: int | None = None,
    ) -> None:
        self.records = JsonlMolecularDataset(
            records_path,
            split=split,
            train_fraction=train_fraction,
            max_records=max_records,
        )
        self.labels = RAScoreLabelStore(labels_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[TrainingRecord, Tensor]:
        record = self.records[index]
        try:
            score = self.labels.get(_smiles(record))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"missing RA label at record index {index}") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"RA score must be finite and in [0,1], got {score}")
        return record, torch.tensor(score, dtype=torch.float32)


def collate_ra_records(batch: list[tuple[TrainingRecord, Tensor]]) -> tuple[list[TrainingRecord], Tensor]:
    if not batch:
        raise ValueError("RA reward batch must not be empty")
    records, targets = zip(*batch)
    return list(records), torch.stack(list(targets))


def load_score_callable(import_path: str) -> Callable[[str], float]:
    """Load a user-supplied RAscore teacher as ``module:function``."""
    if not isinstance(import_path, str) or ":" not in import_path:
        raise ConfigurationError("score_function must use the form module:function")
    module_name, function_name = import_path.split(":", 1)
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(f"cannot load RAscore function {import_path}") from exc
    if not callable(function):
        raise ConfigurationError(f"RAscore function is not callable: {import_path}")
    return function


def build_rascore_label_cache(
    records_path: str | Path,
    output_path: str | Path,
    score_function: Callable[[str], float],
    *,
    overwrite: bool = False,
    teacher_version: str = "external_rascore",
) -> dict[str, object]:
    """Evaluate a teacher once per unique SMILES and persist its labels."""
    records_file = Path(records_path)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"RA label cache exists; pass overwrite=True: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    dataset = JsonlMolecularDataset(records_file)
    written = failed = 0
    seen: set[str] = set()
    with sqlite3.connect(str(output)) as connection:
        connection.execute(
            """
            CREATE TABLE rascore_labels (
                canonical_smiles TEXT PRIMARY KEY,
                ra_score REAL NOT NULL,
                teacher_version TEXT NOT NULL
            )
            """
        )
        for index in range(len(dataset)):
            record = dataset[index]
            smiles = _smiles(record)
            if smiles in seen:
                continue
            seen.add(smiles)
            try:
                score = float(score_function(smiles))
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"teacher returned invalid score {score}")
                connection.execute(
                    "INSERT INTO rascore_labels VALUES (?, ?, ?)",
                    (smiles, score, teacher_version),
                )
                written += 1
            except Exception:
                failed += 1
        connection.commit()
    return {
        "unique_smiles": len(seen),
        "written": written,
        "failed": failed,
        "output_path": str(output),
        "teacher_version": teacher_version,
    }


__all__ = [
    "RAScoreLabelStore",
    "RAScoreLabeledDataset",
    "build_rascore_label_cache",
    "collate_ra_records",
    "load_score_callable",
]
