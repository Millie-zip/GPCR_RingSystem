"""Conservative standard-library logging configuration."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for command-line entrypoints."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
