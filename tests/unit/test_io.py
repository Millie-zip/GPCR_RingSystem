"""Tests for safe directory creation and JSON output."""

import json

from gpcr_ringgen.core.types import GeneratedMolecule
from gpcr_ringgen.utils.io import ensure_directory, save_json


def test_basic_io_writes_inspectable_json(tmp_path) -> None:
    directory = ensure_directory(tmp_path / "nested")
    output = save_json(GeneratedMolecule("dummy"), directory / "result.json")
    assert json.loads(output.read_text(encoding="utf-8"))["molecule_id"] == "dummy"
