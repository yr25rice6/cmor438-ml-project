# CMOR 438 / INDE 577 Machine Learning Project

## Project Overview

This is the public repository for the CMOR 438 / INDE 577 final project. The
project will explore a range of machine learning algorithms, applying them to a
dataset and documenting the data exploration, modeling, evaluation, and
interpretation process. A small custom Python package (`cmor438_ml`) will hold
reusable code that supports the analysis.

## Current Status

**Initial setup.** The repository currently contains only the project skeleton:
directory structure, packaging configuration, environment setup, and pytest
smoke tests for the package. The machine learning algorithms, notebooks,
experiments, and results have not been added yet and will be developed in later
stages of the project.

## Planned Repository Structure

```
.
├── docs/                 # Project scope and supporting documentation
├── notebooks/            # Jupyter notebooks (exploration, modeling, evaluation)
├── examples/             # Usage examples for the custom package
├── data/
│   ├── raw/              # Original, unmodified data (not tracked)
│   └── processed/        # Cleaned/derived data (not tracked)
├── reports/
│   └── figures/          # Generated figures and plots
├── src/
│   └── cmor438_ml/       # Custom Python package
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Build and tooling configuration
```

## Setup

Create and activate a virtual environment, then install the dependencies and the
package in editable mode.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Testing

Run the smoke tests to confirm the package is installed and importable:

```powershell
python -m pytest
```

## Course Alignment

This project is the final deliverable for CMOR 438 / INDE 577. It is intended to
demonstrate understanding of the machine learning algorithms covered in the
course through a complete, reproducible analysis. See `docs/project_scope.md`
for the detailed scope and deliverables.
