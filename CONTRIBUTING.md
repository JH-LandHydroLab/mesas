# Contributing to MESAS

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/JH-LandHydroLab/mesas.git
   cd mesas
   ```

2. Create a conda environment with the required dependencies:
   ```bash
   conda create -n mesas-dev python=3.11 numpy">=1.22,<2" scipy pandas matplotlib numba pytest pytest-cov pre-commit ruff -c conda-forge
   conda activate mesas-dev
   ```

3. Install the package in editable mode (no compiler needed -- the solver is pure Python, JIT-compiled with Numba):
   ```bash
   pip install -e .
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
pytest test/ -v
```

With coverage:
```bash
pytest test/ --cov=mesas --cov-report=term-missing
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks enforce this automatically on every commit. To run manually:

```bash
ruff check --fix mesas/
ruff format mesas/
```

## Project Structure

- `mesas/sas/` - Core SAS transport model
  - `model.py` - `Model` class (main entry point)
  - `specs.py` - SAS specification and component classes
  - `functions.py` - SAS function implementations (piecewise, continuous)
  - `_solve_numba.py` - Numba-accelerated solver (characteristic method with RK integration)
- `mesas/me/` - Model estimation (recursive splitting)
- `mesas/utils/` - Visualization utilities
- `test/` - Test suite
- `examples/` - Bundled example datasets and configs
