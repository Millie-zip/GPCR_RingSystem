"""Build an independent 3D asset database from the read-only source DB."""

from __future__ import annotations

import argparse
from pathlib import Path

from gpcr_ringgen.preprocessing.diffusion_dataset import (
    ThreeDAssetBuilderConfig,
    build_3d_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0xF00D)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--molecule-table", default="selected_molecules")
    parser.add_argument("--ring-table", default="selected_ring_systems")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_3d_assets(
        ThreeDAssetBuilderConfig(
            source_db=args.source_db,
            output_db=args.output_db,
            molecule_table=args.molecule_table,
            ring_table=args.ring_table,
            seed=args.seed,
            limit=args.limit,
            resume=args.resume,
        )
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
