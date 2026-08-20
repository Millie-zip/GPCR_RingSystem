"""Run pocket-disabled inference from one materialized training record."""

from __future__ import annotations

import argparse

from gpcr_ringgen.training.inference import run_record_inference
from gpcr_ringgen.utils.io import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference YAML configuration")
    args = parser.parse_args()
    print(f"Inference output: {run_record_inference(load_yaml_config(args.config)).resolve()}")


if __name__ == "__main__":
    main()
