"""Build an independent 3D asset DB and optionally materialize training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpcr_ringgen.data.three_d_assets import ThreeDAssetBuilderConfig, build_3d_assets
from gpcr_ringgen.data.materialize import materialize_training_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--training-jsonl", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("configs/data/stage3_feature_manifest.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0xF00D)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-jsonl", action="store_true")
    args = parser.parse_args()
    result = build_3d_assets(ThreeDAssetBuilderConfig(
        source_db=args.source_db,
        output_db=args.output_db,
        seed=args.seed,
        limit=args.limit,
        resume=args.resume,
        progress_path=args.progress,
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.training_jsonl is not None:
        materialized = materialize_training_jsonl(
            args.output_db,
            args.training_jsonl,
            args.manifest,
            limit=args.limit,
            overwrite=args.overwrite_jsonl,
            summary_path=args.training_jsonl.with_suffix(args.training_jsonl.suffix + ".summary.json"),
        )
        print(json.dumps(materialized, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
