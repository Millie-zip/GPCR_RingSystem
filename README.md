# gpcr-ringgen

`gpcr-ringgen` is a research codebase scaffold for future GPCR ring-system molecular generation work.

> Current status:  
> Project skeleton only.  
> Scientific modules are placeholders.

No ring-processing, diffusion, equivariant-network, pocket-guidance, docking, or other scientific algorithm is implemented yet. The only executable workflow is a deterministic Dummy Pipeline used to verify software boundaries.

## Environment requirements

- Recommended Python version: Python 3.10
- Minimum supported Python version: Python 3.10

Python 3.10 is the default development and test environment. Python 3.11+ is not artificially blocked, but users of other versions must confirm dependency compatibility in their own environment.

## First-time setup

Linux/macOS:

    git clone <repository>
    cd gpcr-ringgen
    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .

Windows activation:

    .venv\Scripts\activate

For reproducible automation, commands may directly use `.venv/bin/python` on Linux/macOS or `.venv\Scripts\python.exe` on Windows instead of relying on activation.

## Subsequent usage

Do not recreate `.venv` when returning to the project. Reactivate the existing environment:

    source .venv/bin/activate
    python scripts/validate_environment.py
    python -m pytest

Or run without activation:

    .venv/bin/python scripts/validate_environment.py
    .venv/bin/python -m pytest

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Common commands

    python scripts/validate_environment.py
    python -m pytest
    python scripts/sample.py --config configs/experiment/dummy.yaml

The Dummy Pipeline writes an inspectable placeholder result to `outputs/dummy_sample.json`. It does not generate a chemically meaningful molecule.
