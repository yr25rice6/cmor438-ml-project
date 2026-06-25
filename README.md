# CMOR 438 / INDE 577 Machine Learning Project

## Project Overview

This repository contains the final project for CMOR 438 / INDE 577. The project
will explore a range of machine learning algorithms, applying them to a dataset
and documenting the data exploration, modeling, evaluation, and interpretation
process. A small custom Python package (`cmor438_ml`) will hold reusable code
that supports the analysis.

## Current Status

**Initial setup.** The repository currently contains only the project skeleton:
directory structure, packaging configuration, and a minimal smoke test. The
machine learning algorithms, notebooks, experiments, and results have not been
added yet and will be developed in later stages of the project.

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

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Testing

Run the test suite with:

```bash
python -m pytest
```

## Course Alignment

This project is the final deliverable for CMOR 438 / INDE 577. It is intended to
demonstrate understanding of the machine learning algorithms covered in the
course through a complete, reproducible analysis. See `docs/project_scope.md`
for the detailed scope and deliverables.
