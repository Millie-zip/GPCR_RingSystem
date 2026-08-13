# Development guide

- Python 3.10 is the recommended development and test environment.
- Package metadata supports Python >= 3.10 and does not impose an artificial upper bound.
- Use the existing Conda environment named `GPCR`; do not recreate it during project development.
- The package uses a `src` layout and must be installed in editable mode for development.
- Use type hints and concise docstrings for public boundaries.
- Run tests with `python -m pytest` from the project environment.
- Respect module ownership documented in `architecture.md`.
- Avoid circular imports and keep canonical collaboration types in `core`.
- Do not commit datasets, model checkpoints, secrets, logs, or generated outputs.
- Scripts are entrypoints only; scientific logic belongs inside package modules.
- Do not silently turn a placeholder into a scientific design decision. Record and review choices about diffusion, model architecture, vocabularies, graph/pocket representations, losses, guidance, sampling, and rollback before implementation.

Recommended validation sequence:

    conda activate GPCR
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .
    python scripts/validate_environment.py
    python -m pytest
    python scripts/sample.py --config configs/experiment/dummy.yaml
