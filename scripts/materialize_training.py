"""Materialize an audited 3D asset database into Stage 3 JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpcr_ringgen.data.materialize import materialize_training_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-db", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = materialize_training_jsonl(
        args.asset_db,
        args.output_jsonl,
        args.manifest,
        limit=args.limit,
        overwrite=args.overwrite,
        summary_path=args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
