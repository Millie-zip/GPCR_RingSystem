"""Tests for YAML configuration loading."""

import pytest

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.utils.io import load_yaml_config


def test_load_yaml_config(dummy_config_path) -> None:
    config = load_yaml_config(dummy_config_path)
    assert config["experiment"]["name"] == "fixture-dummy"


def test_load_yaml_config_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_yaml_config(tmp_path / "missing.yaml")
