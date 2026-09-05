"""Entrypoint for future training orchestration."""

import argparse
import logging

from gpcr_ringgen.training.trainer import run_training
from gpcr_ringgen.utils.io import load_yaml_config
from gpcr_ringgen.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the YAML configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load configuration and delegate to the training package."""
    args = parse_args()
    configure_logging()
    LOGGER.info("Loading training config: %s", args.config)
    result = run_training(load_yaml_config(args.config))
    LOGGER.info("Training command completed successfully")
    print(f"Training complete: {result.output_dir.resolve()}")
    print(f"Last checkpoint: {result.last_checkpoint.resolve()}")
    print(f"Best checkpoint: {result.best_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
