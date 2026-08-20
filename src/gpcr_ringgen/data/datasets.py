"""Dataset interfaces and random-access JSONL reader for real assets."""

from abc import ABC, abstractmethod
from array import array
import json
from pathlib import Path
from typing import Any

from gpcr_ringgen.data.records import TrainingRecord


class MolecularDataset(ABC):
    """Minimal dataset boundary for future implementations."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of records."""

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return one record without prescribing its final schema."""


class JsonlMolecularDataset(MolecularDataset):
    """Random-access loader with a byte-offset sidecar."""

    def __init__(self, path: str | Path, *, max_records: int | None = None, split: str | None = None, train_fraction: float = 0.9) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"training JSONL not found: {self.path}")
        self.index_path = self.path.with_suffix(self.path.suffix + ".idx")
        offsets = self._read_offsets() or self._build_offsets()
        if split is not None:
            if split not in {"train", "validation"}:
                raise ValueError("split must be train, validation, or None")
            if not 0.0 < train_fraction < 1.0:
                raise ValueError("train_fraction must be between 0 and 1")
            offsets = self._filter_split(offsets, split, train_fraction)
        if max_records is not None:
            if not isinstance(max_records, int) or max_records <= 0:
                raise ValueError("max_records must be a positive integer")
            offsets = offsets[:max_records]
        if not offsets:
            raise ValueError(f"training JSONL is empty: {self.path}")
        self._offsets = offsets
        self._handle = None

    def _filter_split(self, offsets: list[int], split: str, train_fraction: float) -> list[int]:
        import hashlib

        selected: list[int] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for offset in offsets:
                handle.seek(offset)
                payload = json.loads(handle.readline())
                metadata = payload.get("metadata", {})
                group = str(metadata.get("split_group", payload.get("sample_id", offset)))
                bucket = int.from_bytes(hashlib.sha256(group.encode("utf-8")).digest()[:8], "big") / 2**64
                is_train = bucket < train_fraction
                if (split == "train" and is_train) or (split == "validation" and not is_train):
                    selected.append(offset)
        return selected

    def _read_offsets(self) -> list[int] | None:
        try:
            if not self.index_path.is_file():
                return None
            values = array("Q")
            with self.index_path.open("rb") as handle:
                values.fromfile(handle, self.index_path.stat().st_size // values.itemsize)
            if values and values[-1] >= self.path.stat().st_size:
                return None
            return list(values)
        except (OSError, EOFError, ValueError):
            return None

    def _build_offsets(self) -> list[int]:
        values = array("Q")
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    values.append(offset)
        try:
            with self.index_path.open("wb") as handle:
                values.tofile(handle)
        except OSError:
            pass
        return list(values)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> TrainingRecord:
        if not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("r", encoding="utf-8")
        self._handle.seek(self._offsets[index])
        try:
            payload = json.loads(self._handle.readline())
            return TrainingRecord.from_mapping(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid training record at index {index}") from exc

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()


__all__ = ["JsonlMolecularDataset", "MolecularDataset"]
