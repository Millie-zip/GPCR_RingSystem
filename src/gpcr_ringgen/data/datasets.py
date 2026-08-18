"""Dataset interfaces and the versioned JSONL Stage 3 asset loader."""

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
    """Random-access reader for the converter's JSONL training asset library.

    A compact byte-offset sidecar is created next to the JSONL file on first
    use.  The source JSONL remains immutable from the loader's perspective and
    can therefore be kept on a read-only data volume after the sidecar exists.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_records: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"training asset JSONL not found: {self.path}")
        if max_records is not None and (
            isinstance(max_records, bool) or not isinstance(max_records, int)
        ):
            raise TypeError("max_records must be an integer or None")
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive when provided")
        self.index_path = self.path.with_suffix(self.path.suffix + ".idx")
        offsets = self._load_index()
        if offsets is None:
            offsets = self._build_index()
        if max_records is not None:
            offsets = offsets[:max_records]
        if not offsets:
            raise ValueError(f"training asset JSONL is empty: {self.path}")
        self._offsets = offsets
        self._handle = None

    def _load_index(self) -> list[int] | None:
        if not self.index_path.is_file():
            return None
        try:
            values = array("Q")
            with self.index_path.open("rb") as handle:
                values.fromfile(handle, self.index_path.stat().st_size // values.itemsize)
            if values and values[-1] >= self.path.stat().st_size:
                return None
            return list(values)
        except (OSError, EOFError, ValueError):
            return None

    def _build_index(self) -> list[int]:
        offsets = array("Q")
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        try:
            with self.index_path.open("wb") as handle:
                offsets.tofile(handle)
        except OSError:
            # A read-only data volume is still usable; only the cache is lost.
            pass
        return list(offsets)

    def _file(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> TrainingRecord:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        handle = self._file()
        handle.seek(self._offsets[index])
        line = handle.readline()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record at index {index}") from exc
        try:
            return TrainingRecord.from_mapping(payload)
        except (TypeError, ValueError) as exc:
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
