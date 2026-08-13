"""Run the independent Stage 3 pocket-conditioned inference pipeline."""

from __future__ import annotations

import argparse
import logging

from gpcr_ringgen.pipeline import run_stage3_inference_from_config
from gpcr_ringgen.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the unified Stage 3 pipeline configuration path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    """Execute Stage 3 inference and report the structured output path."""

    args = parse_args()
    configure_logging()
    result = run_stage3_inference_from_config(args.config)
    if result.random_weights:
        LOGGER.warning(
            "No checkpoint was loaded. Output is an untrained pipeline smoke test "
            "and is not a valid molecular or pharmacological result."
        )
    LOGGER.info("Stage 3 inference completed; output=%s", result.output_path.resolve())


if __name__ == "__main__":
    main()
