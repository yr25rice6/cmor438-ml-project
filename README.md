# CMOR 438 / INDE 577 Machine Learning Project

## Project Overview

This repository implements a reproducible binary classification workflow for the
Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to classify tumors as
malignant or benign while keeping the analysis clean and repeatable. To support
the workflow, the project includes a small custom Python package (`cmor438_ml`)
that provides reusable machine learning utilities and a couple of from-scratch
models, so the notebook stays focused on the analysis rather than repeating
boilerplate code.

## Repository Structure

```
.
├── src/
│   └── cmor438_ml/       # Custom Python package (utilities and helpers)
│       └── models/       # From-scratch model implementations
├── tests/                # Test suite for the package
├── notebooks/            # Jupyter notebooks (main workflow + per-algorithm)
│   ├── Supervised/       # Dedicated supervised-algorithm notebooks
│   └── Unsupervised/     # Dedicated unsupervised-algorithm notebooks
├── docs/                 # Project scope, planning, and supporting notes
├── data/                 # Local data directory (raw/processed, not tracked)
├── reports/
│   └── figures/          # Generated figures and plots
├── examples/             # Standalone usage examples for the custom package
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Build and tooling configuration
```

- **`src/cmor438_ml/`** — the custom package holding reusable utilities for
  validation, metrics, preprocessing, dataset loading, and evaluation.
- **`src/cmor438_ml/models/`** — from-scratch model implementations
  (`KNNClassifier` and `LogisticRegressionGD`).
- **`tests/`** — pytest suite covering the package behavior.
- **`notebooks/`** — the main analysis notebook plus dedicated per-algorithm
  notebooks organized into `Supervised/` and `Unsupervised/` directories.
- **`docs/`** — project scope, planning notes, and dataset/algorithm decisions.
- **`data/`** — local directory for raw and processed data (not tracked in git).
- **`reports/figures/`** — output directory for generated figures and plots.
- **`examples/`** — standalone usage examples for the custom package.

## Implemented Package Components

The `cmor438_ml` package currently provides:

- **Validation helpers** — input checks such as array/shape validation, fitted
  estimator checks, and binary label verification.
- **Classification metrics** — accuracy, confusion matrix, precision, recall,
  F1 score, and ROC/AUC helpers (`roc_curve_binary` and `auc_score`).
- **Preprocessing utilities** — a train/test split, a from-scratch standard
  scaler (`StandardScalerScratch`), and an intercept helper.
- **Dataset loader** — `load_breast_cancer_data` for loading the Breast Cancer
  Wisconsin dataset.
- **Evaluation utilities** — functions to evaluate a classifier, build a
  classification report, and compare classifiers.
- **`KNNClassifier`** — a from-scratch k-nearest-neighbors classifier.
- **`LogisticRegressionGD`** — a from-scratch logistic regression model trained
  with gradient descent.

## Notebook

The main analysis lives in
`notebooks/01_breast_cancer_classification_workflow.ipynb`. It walks through the
end-to-end workflow and demonstrates:

- loading the Breast Cancer Wisconsin dataset
- a brief exploratory analysis of the data
- a train/test split
- standardization of the features
- training both the KNN and logistic regression models
- evaluating each model with the package metrics
- comparing the two models
- a five-fold cross-validation check to gauge how sensitive the comparison is to
  the particular train/test split
- a brief KNN hyperparameter check (varying the number of neighbors)
- a ROC/AUC analysis of the logistic regression model that evaluates its ranking
  quality across thresholds (treating benign as the positive class)

The `notebooks/` directory is also organized into `Supervised/` and
`Unsupervised/` subdirectories that hold dedicated per-algorithm notebooks
following the structure of the course example repositories. The repository now
includes dedicated notebooks for **Perceptron**, **Gradient Descent**, and
**Linear Regression**, along with **Logistic Regression**, **Multilayer
Perceptron**, and **K Nearest Neighbors**; the remaining algorithm directories
are placeholders that will be filled in a later step. See `notebooks/README.md`
for the full index.

## Examples

The `examples/` directory holds standalone scripts that use the `cmor438_ml`
package outside of the notebook. The main one is
`examples/basic_classification_demo.py`, a lightweight command-line example that
loads the dataset, standardizes the features, trains both models, and prints a
short summary of their test-set metrics (including a ROC/AUC summary for the
logistic regression model). It produces no plots and writes no files.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe examples/basic_classification_demo.py
```

See `examples/README.md` for more detail.

## Installation

Create and activate a virtual environment, then install the dependencies and the
package in editable mode.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Running Tests

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current suite includes tests for validation, metrics, preprocessing,
datasets, evaluation, KNN, logistic regression, and package imports.

## Running the Notebook

The notebook can be executed from the command line:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_breast_cancer_classification_workflow.ipynb
```

It can also be opened and run interactively in Jupyter (for example with
`jupyter notebook` or `jupyter lab`).

## Reproducibility Notes

- Fixed random states are used where appropriate (for example in the train/test
  split) so that results are repeatable.
- The dataset is bundled with scikit-learn, so no manual download is required.
- Reusable package functions reduce duplicated code in the notebook and keep the
  analysis consistent.
- Tests are included for the core package behavior to help catch regressions.

## Limitations and Next Steps

- The notebook reports both a single train/test split and a simple k-fold
  cross-validation check (a `cross_validate_classifier` utility has been added to
  the package for this).
- The cross-validation check standardizes the feature matrix once up front rather
  than refitting preprocessing inside each fold.
- ROC/AUC utilities (`roc_curve_binary` and `auc_score`) have been added to the
  package, and the notebook now includes a ROC/AUC demonstration for the logistic
  regression model.
- Future work may include stricter per-fold preprocessing pipelines, additional
  models, and further notebooks to extend the comparison.
