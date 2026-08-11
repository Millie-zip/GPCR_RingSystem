# Development guide

- Python 3.10 is the recommended development and test environment.
- Package metadata supports Python >= 3.10 and does not impose an artificial upper bound.
- Use the project `.venv`; create it only on first setup and never recreate it on every run.
- The package uses a `src` layout and must be installed in editable mode for development.
- Use type hints and concise docstrings for public boundaries.
- Run tests with `python -m pytest` from the project environment.
- Respect module ownership documented in `architecture.md`.
- Avoid circular imports and keep canonical collaboration types in `core`.
- Do not commit datasets, model checkpoints, secrets, logs, or generated outputs.
- Scripts are entrypoints only; scientific logic belongs inside package modules.
- Do not silently turn a placeholder into a scientific design decision. Record and review choices about diffusion, model architecture, vocabularies, graph/pocket representations, losses, guidance, sampling, and rollback before implementation.

Recommended validation sequence:

    .venv/bin/python -m pip install -e .
    .venv/bin/python scripts/validate_environment.py
    .venv/bin/python -m pytest
    .venv/bin/python scripts/sample.py --config configs/experiment/dummy.yaml

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
