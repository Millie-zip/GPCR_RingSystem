"""Basic configuration and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from gpcr_ringgen.core.exceptions import ConfigurationError


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if necessary and return its resolved path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk with clear validation errors."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping: {config_path}"
        )
    return loaded


def save_json(data: Any, path: str | Path) -> Path:
    """Serialize JSON-compatible data or a dataclass to an inspectable file."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    serializable = asdict(data) if is_dataclass(data) else data
    output_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
