# Implementation Plan

This document specifies the planned implementation design for the first real
code phase of the CMOR 438 / INDE 577 final project. It builds on the confirmed
decisions in `dataset_algorithm_decision.md` and refines them into a concrete
package, API, and testing design. It is a planning document only: no algorithms
have been implemented, no package modules beyond the existing skeleton have been
created, no notebooks exist, no datasets have been downloaded, and no
experiments or results exist yet. Every file named below as "planned" or "future"
does not exist at this stage.

## 1. Implementation Objective

The first implementation phase will build reusable package components for the
**Breast Cancer Wisconsin (Diagnostic)** classification workflow. These
components form the shared foundation that the later notebooks and model
comparisons will depend on.

The goal of this phase is **correctness, testing, and readable API design before
advanced modeling**. The phase delivers a small set of well-understood,
thoroughly tested utilities and classifiers rather than breadth. Advanced
modeling (ensembles, SVM, neural networks) and the unsupervised demonstrations
are explicitly out of scope until this core package is reliable.

## 2. Planned Package Structure

The proposed future structure is shown below. **These files are not created
yet**; only `src/cmor438_ml/__init__.py` currently exists. The structure is a
target for this phase, to be built incrementally in the order given in
Section 6.

```
src/cmor438_ml/
  __init__.py
  preprocessing.py
  metrics.py
  validation.py
  models/
    __init__.py
    logistic_regression.py
    knn.py
```

## 3. Module Responsibilities

The planned responsibility of each future module:

- **`preprocessing.py`** — Data preparation helpers: train/test split,
  standardization helpers, and an optional add-intercept utility for models that
  expect a bias column.
- **`metrics.py`** — Classification metrics: accuracy, precision, recall, F1,
  and confusion matrix.
- **`validation.py`** — Shared input-handling helpers: array shape checks,
  fitted-state checks (verifying an estimator has been fit before predicting),
  and input conversion helpers (coercing inputs to consistent NumPy arrays).
- **`models/logistic_regression.py`** — A binary logistic regression classifier
  trained with gradient descent.
- **`models/knn.py`** — A k-nearest neighbors classifier.

## 4. Planned Public APIs

The planned function and class names with high-level signatures. Exact
parameters may be refined during implementation, but the public surface is
intended to stay consistent with scikit-learn conventions where practical.

### `preprocessing.py`

- `train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True)` —
  Split arrays into train and test partitions with optional deterministic
  shuffling.
- `StandardScalerScratch` — Standardization transformer with:
  - `fit(X)` — compute and store per-feature mean and standard deviation.
  - `transform(X)` — apply the stored standardization.
  - `fit_transform(X)` — fit then transform in one call.
- `add_intercept(X)` (optional utility) — prepend a column of ones for models
  that include a bias term.

### `metrics.py`

- `accuracy_score(y_true, y_pred)`
- `precision_score(y_true, y_pred)`
- `recall_score(y_true, y_pred)`
- `f1_score(y_true, y_pred)`
- `confusion_matrix(y_true, y_pred)`

### `models/logistic_regression.py`

- `LogisticRegressionGD(learning_rate=0.01, n_iterations=1000, random_state=None)`
  with:
  - `fit(X, y)` — train via gradient descent; returns `self`.
  - `predict_proba(X)` — return predicted class-1 probabilities.
  - `predict(X)` — return predicted class labels.
  - `score(X, y)` — return accuracy on the given data.

### `models/knn.py`

- `KNNClassifier(n_neighbors=5)` with:
  - `fit(X, y)` — store the training data; returns `self`.
  - `predict(X)` — return predicted class labels by majority vote.
  - `score(X, y)` — return accuracy on the given data.

## 5. Testing Strategy

The first pytest files to create later, alongside the modules they cover:

- `tests/test_metrics.py`
- `tests/test_preprocessing.py`
- `tests/test_validation.py`
- `tests/test_logistic_regression.py`
- `tests/test_knn.py`

What each should test:

- **`test_metrics.py`** — Hand-computed metric examples on small, fixed inputs,
  so each metric (accuracy, precision, recall, F1, confusion matrix) matches a
  value worked out by hand.
- **`test_preprocessing.py`** — Deterministic train/test split behavior under a
  fixed `random_state` (same partition across runs, correct sizes, no overlap),
  and standardization mean/std behavior (transformed features have approximately
  zero mean and unit standard deviation).
- **`test_validation.py`** — Shape validation and error handling: mismatched
  shapes, empty inputs, and predicting before fitting raise clear, expected
  errors; input conversion produces consistent array types.
- **`test_logistic_regression.py`** — Model fit/predict output shape,
  reproducibility where `random_state` applies, and a simple separable example
  for a classifier sanity check (the model learns to separate clearly separable
  classes).
- **`test_knn.py`** — Model fit/predict output shape and a simple separable
  example for a sanity check (neighbors of clearly separable points yield the
  correct labels).

Tests compare against simple, defensible expected behavior and hand-computed
values, not invented or assumed performance numbers.

## 6. Implementation Order

The recommended build order moves from shared foundations outward to models,
then exports, then notebooks:

1. **Validation helpers** (`validation.py`) — needed by everything else.
2. **Metrics** (`metrics.py`) — depends only on validation and NumPy.
3. **Preprocessing** (`preprocessing.py`) — split and standardization helpers.
4. **KNN classifier** (`models/knn.py`) — the simpler model first.
5. **Logistic regression classifier** (`models/logistic_regression.py`).
6. **Update `__init__.py` exports** — only after the modules above exist and
   their tests pass.
7. **Notebooks** — created only after the package tests pass.

## 7. Design Rules

- Keep implementations simple and readable.
- Use NumPy for core logic.
- Avoid unnecessary dependencies inside package code.
- Keep APIs consistent with scikit-learn conventions where practical
  (`fit` / `transform` / `predict` / `score`, `random_state`).
- Write tests before or alongside each implementation.
- Do not optimize prematurely.
- Do not add neural networks, SVM, or ensemble code until the core package is
  reliable.

## 8. Definition of Done for the First Code Phase

The first code phase is complete when:

- All planned modules (`validation.py`, `metrics.py`, `preprocessing.py`,
  `models/knn.py`, `models/logistic_regression.py`, and the `models` package)
  exist.
- All first-pass tests pass.
- The package imports cleanly from the virtual environment.
- No notebooks depend on untested package code.
- README usage instructions remain accurate.
