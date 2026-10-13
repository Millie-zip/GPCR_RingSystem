"""One-molecule JSON worker for bounded notebook conformer generation."""

from __future__ import annotations

import json
import sys

from gpcr_ringgen.preprocessing.diffusion_dataset.notebook_conformer import (
    generate_notebook_conformer,
)


def main() -> None:
    request = json.load(sys.stdin)
    result = generate_notebook_conformer(
        request["smiles"],
        request["atoms"],
        request["bonds"],
        random_seed=int(request.get("random_seed", 0xF00D)),
    )
    json.dump(result, sys.stdout, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
