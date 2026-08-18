"""Train the Stage 3 model from a converted real-data JSONL asset library."""

import argparse

from gpcr_ringgen.training.trainer import run_training
from gpcr_ringgen.utils.io import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse the YAML configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load configuration, train, and print the artifact paths."""
    args = parse_args()
    result = run_training(load_yaml_config(args.config))
    print(f"Training complete: {result.output_dir.resolve()}")
    print(f"Last checkpoint: {result.last_checkpoint.resolve()}")
    print(f"Best checkpoint: {result.best_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
