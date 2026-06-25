# Project Plan

This document plans the CMOR 438 / INDE 577 final project. It proposes
candidate project directions, recommends one, and outlines the algorithm,
notebook, package, and testing work needed to deliver a strong submission. It is
a planning document only: no datasets have been downloaded, no algorithms have
been implemented, and no experiments or results exist yet. All items below are
proposals to be confirmed before implementation.

## 1. Project Direction

Three realistic directions are proposed. Each favors tabular data and runs
comfortably on a laptop. The source and license of any candidate dataset must be
verified before implementation (see Section 8).

- **Direction A — Wine Quality (classification + regression).** The UCI Wine
  Quality dataset relates physicochemical measurements to a quality score. The
  score supports regression directly and can be binned into classes
  (e.g., low/high quality) for classification. Modest size and all-numeric
  features make preprocessing straightforward.

- **Direction B — Breast Cancer Wisconsin (classification + unsupervised).** A
  clean, all-numeric diagnostic dataset (benign vs. malignant). It supports
  binary classification and, because features are continuous and correlated, is
  also a natural fit for dimensionality reduction (PCA) and clustering. This is
  the option that most clearly supports unsupervised learning in addition to
  supervised work, and it is available directly through scikit-learn.

- **Direction C — California Housing (regression).** A regression-focused
  dataset predicting median house value from socioeconomic and geographic
  features. Larger and noisier than the others, with useful feature-engineering
  and scaling challenges, but weaker for classification and unsupervised work.

## 2. Recommended Direction

**Recommended: Direction B — Breast Cancer Wisconsin**, with a secondary
regression demonstration drawn from a built-in regression dataset if a
regression component is desired.

This direction is the most practical and best aligned with the rubric because:

- **Data preprocessing.** All-numeric features with differing scales motivate a
  clear, demonstrable preprocessing pipeline (train/test split, standardization,
  optional feature selection) without the burden of messy text or missing-value
  handling.
- **Supervised learning.** Binary classification gives a clean, interpretable
  target and a well-understood set of evaluation metrics (accuracy, precision,
  recall, F1, ROC-AUC).
- **Model evaluation.** Two balanced classes and a moderate feature count make
  cross-validation, confusion matrices, and ROC curves easy to compute and
  interpret.
- **Package usage.** The workflow naturally factors into reusable pieces
  (metrics, preprocessing helpers, model wrappers) that justify a custom
  package rather than ad hoc notebook code.
- **Tests.** Deterministic, small, numeric data makes correctness and
  reproducibility tests practical to write.
- **Documentation.** A single coherent dataset and target keeps the narrative
  focused and easy to explain.

The dataset is available through scikit-learn's loaders, so the project can
proceed without external downloads while remaining a real, citable dataset.
Scope is kept deliberately modest: depth, correctness, and clarity are preferred
over breadth.

## 3. Algorithm Scope

The emphasis is correctness, testing, and interpretability over breadth. A small
set of well-understood algorithms is preferred.

### Custom package implementations

- Logistic regression (gradient descent), used as the primary from-scratch
  classifier.
- K-nearest neighbors classifier, as a simple, transparent comparison.
- A core set of metrics (accuracy, precision, recall, F1, confusion matrix) and
  preprocessing helpers (train/test split, standardization).

### scikit-learn comparison baselines

- `LogisticRegression` and `KNeighborsClassifier`, to validate the custom
  implementations against trusted references.
- One or two stronger baselines (e.g., `DecisionTreeClassifier`,
  `RandomForestClassifier`) to contextualize performance.

### Notebook-only demonstrations

- Principal Component Analysis (PCA) for dimensionality reduction and
  visualization.
- K-means clustering, compared informally against the known diagnostic labels.
- Optionally, a brief regression demonstration on a built-in regression dataset
  to round out supervised coverage.

## 4. Proposed Notebook Plan

A small, focused set of notebooks, each with one clear purpose:

1. **`01_data_exploration.ipynb`** — Load the dataset, summarize features and
   target balance, visualize distributions and correlations, and define the
   preprocessing approach.
2. **`02_supervised_workflow.ipynb`** — End-to-end supervised pipeline:
   preprocessing, training the custom classifier, and basic evaluation.
3. **`03_model_comparison.ipynb`** — Compare the custom implementations against
   scikit-learn baselines using cross-validation and shared metrics; present
   confusion matrices and ROC curves.
4. **`04_unsupervised.ipynb`** — PCA for dimensionality reduction and
   visualization, plus K-means clustering compared against the labels.
5. **`05_package_usage.ipynb`** — A concise example showing how the
   `cmor438_ml` package is imported and used as a library, mirroring the
   `examples/` material.

## 5. Proposed Package Modules

Proposed future modules under `src/cmor438_ml/`. **These are not created yet**
and are subject to confirmation:

- `models/` — custom estimators (e.g., `logistic_regression.py`, `knn.py`) with
  a consistent `fit` / `predict` interface.
- `metrics.py` — classification (and optionally regression) metrics.
- `preprocessing.py` — train/test split, scaling/standardization, and simple
  feature helpers.
- `validation.py` — input checks and array/shape validation shared across
  models and metrics.
- `plotting.py` — reusable plotting helpers (confusion matrix, ROC curve, PCA
  scatter).
- `utils.py` — small shared utilities (e.g., random-seed handling).
- `datasets.py` (optional) — thin loaders/wrappers around the chosen dataset for
  reproducible access.

## 6. Testing Plan

The pytest suite should eventually cover:

- **Import tests.** Each public module and the top-level package import cleanly
  (extending the existing smoke tests).
- **Metrics correctness.** Metric functions match hand-computed values and known
  reference outputs on small fixed inputs.
- **Model fit/predict behavior.** Estimators expose `fit`/`predict`, return
  correctly shaped outputs, and achieve expected accuracy on simple separable
  data.
- **Input validation.** Mismatched shapes, empty inputs, and invalid arguments
  raise clear, expected errors.
- **Reproducibility.** Fixing a random seed yields identical results across
  runs.
- **Edge cases.** Single-class inputs, single-feature inputs, and small-sample
  behavior are handled gracefully.

## 7. Rubric Alignment

| Criterion (weight) | How the plan addresses it |
| --- | --- |
| **Functionality and Implementation (40%)** | Custom logistic regression and KNN with a clear `fit`/`predict` interface, shared metrics and preprocessing, and a complete supervised + unsupervised workflow across the notebooks. |
| **Documentation and Readability (20%)** | README, project scope, this plan, per-notebook narratives, and docstrings on package modules; a single focused dataset keeps the story clear. |
| **Testing and Reliability (20%)** | The testing plan above covers imports, metric correctness, model behavior, validation, reproducibility, and edge cases. |
| **Examples and Usability (10%)** | A dedicated package-usage notebook and `examples/` material show the package used as a library. |
| **Repository Quality (10%)** | The existing `src/` layout, packaging, documented dependencies, and tidy directory structure are maintained as the project grows. |

## 8. Decisions Still Needed

The following must be confirmed before implementation begins:

- **Final dataset.** Confirm Breast Cancer Wisconsin (recommended) or an
  alternative direction, and whether a separate regression dataset is added.
- **Final algorithm list.** Confirm which algorithms are implemented from
  scratch, which are used as scikit-learn baselines, and which are
  demonstration-only.
- **Neural network content.** Decide whether to include any neural network
  material (e.g., a simple from-scratch perceptron or an MLP baseline) or to
  exclude it to keep scope focused.
- **Clustering / dimensionality reduction.** Confirm whether PCA and K-means are
  included as planned or deferred.
- **Dataset source.** Decide whether to rely only on built-in (scikit-learn)
  datasets or to also use an external public dataset, with attention to size,
  source, and license.
