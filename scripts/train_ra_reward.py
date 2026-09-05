"""Train the independent differentiable RA reward model."""

from __future__ import annotations

import argparse

from gpcr_ringgen.training.ra_reward_training import run_ra_reward_training
from gpcr_ringgen.utils.io import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RA reward training YAML")
    args = parser.parse_args()
    result = run_ra_reward_training(load_yaml_config(args.config))
    print(f"RA reward training complete: {result.output_dir.resolve()}")
    print(f"Last checkpoint: {result.last_checkpoint.resolve()}")
    print(f"Best checkpoint: {result.best_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
