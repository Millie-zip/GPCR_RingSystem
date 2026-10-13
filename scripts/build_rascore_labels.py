"""Build a cached RAscore label SQLite database from Stage 3 JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpcr_ringgen.training.ra_reward_teacher import load_rascore_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--score-function",
        required=True,
        help="RAscore teacher callable in module:function form",
    )
    parser.add_argument("--teacher-version", default="external_rascore")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    teacher = load_rascore_teacher(
        args.score_function,
        version=args.teacher_version,
    )
    result = teacher.build_cache(
        str(args.records),
        str(args.output),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
