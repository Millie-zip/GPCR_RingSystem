"""Validate that the project environment supports the software skeleton."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def main() -> int:
    """Run required environment checks and return a process status code."""
    try:
        version = sys.version_info
        if version < (3, 10):
            print(f"ERROR: Python {version.major}.{version.minor} is below 3.10.")
            return 1
        if version[:2] == (3, 10):
            _pass("Python 3.10 detected (recommended environment).")
        else:
            _pass(
                f"Python {version.major}.{version.minor} detected. "
                "Python 3.10 is the recommended development environment."
            )

        print(f"Python executable: {sys.executable}")
        if sys.prefix != sys.base_prefix:
            _pass("Virtual environment is active.")
        else:
            _warn("No virtual environment detected; use the project .venv.")

        package = importlib.import_module("gpcr_ringgen")
        _pass(f"Package import passed: {package.__file__}")

        from gpcr_ringgen.utils.device import resolve_device
        from gpcr_ringgen.utils.io import load_yaml_config, save_json

        config = load_yaml_config("configs/experiment/dummy.yaml")
        _pass(f"Configuration loading passed: {config['experiment']['name']}")

        torch = importlib.import_module("torch")
        _pass(f"PyTorch import passed: {torch.__version__}")

        device = resolve_device(str(config["device"]["type"]))
        _pass(f"Device initialization passed: {device}")

        probe_path = Path("outputs/environment_validation_probe.json")
        save_json({"status": "passed", "device": str(device)}, probe_path)
        json.loads(probe_path.read_text(encoding="utf-8"))
        probe_path.unlink()
        _pass("Basic output writing passed.")
    except Exception as exc:
        print(f"ERROR: Environment validation failed: {exc}")
        return 1

    print("Environment validation passed.")
    print("Python environment passed.")
    print("Package import passed.")
    print("Configuration loading passed.")
    print("Device initialization passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
