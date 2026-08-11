"""Run the architecture-only GPCR-RingGen Dummy Pipeline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.models.diffusion.model import DummyDiffusionModel
from gpcr_ringgen.sampling.sampler import DummySampler
from gpcr_ringgen.scaffold.realization import DummyScaffoldProvider
from gpcr_ringgen.topology.sampling import DummyTopologyProvider
from gpcr_ringgen.utils.device import resolve_device
from gpcr_ringgen.utils.io import load_yaml_config, save_json
from gpcr_ringgen.utils.logging import configure_logging
from gpcr_ringgen.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the YAML configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration field {key!r} must be a mapping.")
    return value


def run_dummy_pipeline(config: dict[str, Any]) -> Path:
    """Compose Dummy components and save human-readable non-scientific output."""
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")
    set_seed(seed)

    device_name = str(_mapping(config, "device").get("type", "auto"))
    device = resolve_device(device_name)
    sampling = _mapping(config, "sampling")
    num_samples = sampling.get("num_samples", 1)

    topology = DummyTopologyProvider().provide()
    condition = DummyScaffoldProvider().provide(topology)
    molecules = DummySampler().sample(
        model=DummyDiffusionModel(),
        condition=condition,
        guidance=None,
        num_samples=num_samples,
    )

    output = _mapping(config, "output")
    directory = Path(str(output.get("directory", "outputs")))
    filename = str(output.get("filename", "dummy_sample.json"))
    if Path(filename).name != filename:
        raise ConfigurationError("output.filename must be a plain file name")
    payload = {
        "pipeline": "Dummy RingTopology -> Dummy ScaffoldCondition -> "
        "Dummy DiffusionState -> Dummy DiffusionModel -> Dummy Sampler -> "
        "GeneratedMolecule",
        "scientific": False,
        "device": str(device),
        "seed": seed,
        "samples": [asdict(molecule) for molecule in molecules],
    }
    return save_json(payload, directory / filename)


def main() -> None:
    """Load configuration and run the in-package Dummy composition."""
    args = parse_args()
    configure_logging()
    output_path = run_dummy_pipeline(load_yaml_config(args.config))
    LOGGER.info("Dummy Pipeline completed; output=%s", output_path.resolve())


if __name__ == "__main__":
    main()
