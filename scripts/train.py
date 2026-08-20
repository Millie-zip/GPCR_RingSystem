"""Train Stage 3 from a materialized real-data JSONL library."""

import argparse

from gpcr_ringgen.training.trainer import run_training
from gpcr_ringgen.utils.io import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse the configuration path only; training details stay in-package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load configuration, train, and print checkpoint paths."""
    args = parse_args()
    result = run_training(load_yaml_config(args.config))
    print(f"Training complete: {result.output_dir.resolve()}")
    print(f"Last checkpoint: {result.last_checkpoint.resolve()}")
    print(f"Best checkpoint: {result.best_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
