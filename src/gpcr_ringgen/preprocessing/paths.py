"""Shared filesystem conventions for offline preprocessing jobs."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "GPCR_RINGGEN_DATA_ROOT"


def data_root() -> Path:
    """Return the external preprocessing data root."""

    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root.joinpath("data").resolve()

