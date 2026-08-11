"""Entrypoint for future training orchestration."""

import argparse

from gpcr_ringgen.training.trainer import run_training
from gpcr_ringgen.utils.io import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse the configuration path only; training details stay in-package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load configuration and delegate to the training package."""
    args = parse_args()
    run_training(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
