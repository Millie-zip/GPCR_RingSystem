"""Isolated RDKit conformer worker for timeout-safe Stage3 preparation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def write_starting_conformers(
    smiles: str,
    output_sdf: Path,
    *,
    max_conformers: int,
    embed_max_iterations: int,
    uff_max_iters: int,
) -> None:
    from rdkit import Chem

    from gpcr_ringgen.preprocessing.pocket_prefilter.pipeline import (
        generate_conformers,
    )

    generated = generate_conformers(
        smiles,
        max_conformers=max_conformers,
        embed_max_iterations=embed_max_iterations,
        uff_max_iters=uff_max_iters,
    )
    molecule = generated["mol3d"]
    writer = Chem.SDWriter(str(output_sdf))
    try:
        for conformer_id in generated["conf_ids"]:
            writer.write(molecule, confId=int(conformer_id))
    finally:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RDKit starting conformers.")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output-sdf", type=Path, required=True)
    parser.add_argument("--max-conformers", type=int, required=True)
    parser.add_argument("--embed-max-iterations", type=int, required=True)
    parser.add_argument("--uff-max-iters", type=int, required=True)
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in ("max_conformers", "embed_max_iterations", "uff_max_iters"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    write_starting_conformers(
        args.smiles,
        args.output_sdf,
        max_conformers=args.max_conformers,
        embed_max_iterations=args.embed_max_iterations,
        uff_max_iters=args.uff_max_iters,
    )


if __name__ == "__main__":
    cli()
