# Dataset and Algorithm Decision

This document records the confirmed dataset and algorithm decisions for the
CMOR 438 / INDE 577 final project. It resolves the open items left in
`project_scope.md` and `project_plan.md` and fixes the scope for the initial
implementation. It is a planning record only: no algorithms have been
implemented, no notebooks have been created, no datasets have been downloaded,
and no experiments or results exist yet.

## 1. Final Main Dataset

The main dataset is the **Breast Cancer Wisconsin (Diagnostic)** dataset.

- It will be accessed through scikit-learn's built-in dataset loader
  (`sklearn.datasets.load_breast_cancer`) during implementation.
- No dataset file has been downloaded or committed at this stage.
- The source and license details will be verified and cited in the notebooks
  before final submission.

This dataset is clean, all-numeric, and modest in size, so it runs comfortably
on a laptop and avoids the burden of messy text or missing-value handling. Its
continuous, correlated features also make it a natural fit for the unsupervised
component described below.

## 2. Main Prediction Task

The main task is **binary classification**: predicting a benign vs. malignant
diagnosis from the numeric features.

This task fits the course project well because it exercises the full set of
skills the rubric rewards:

- **Preprocessing.** Numeric features with differing scales motivate a clear
  pipeline (train/test split, standardization).
- **Supervised learning.** A clean, interpretable binary target supports
  from-scratch and library models with a consistent `fit` / `predict`
  interface.
- **Metrics.** Two reasonably balanced classes make accuracy, precision,
  recall, F1, and confusion matrices easy to compute and interpret.
- **Model comparison.** A moderate feature count keeps cross-validation and
  baseline comparison fast and reproducible.
- **Documentation.** A single coherent dataset and target keeps the project
  narrative focused.
- **Testing.** Small, deterministic, numeric data makes correctness and
  reproducibility tests practical to write.

## 3. Unsupervised / Dimensionality Reduction Component

The project includes an unsupervised component built on the same dataset:

- **PCA** for visualization and dimensionality reduction, used to project the
  high-dimensional features into two or three components for inspection.
- **K-means** clustering as an unsupervised demonstration, compared informally
  against the known diagnostic labels to see how well unsupervised structure
  aligns with the supervised target.

This component is a demonstration of an unsupervised workflow, not a clinical
diagnostic claim. Any comparison against the diagnostic labels is illustrative
only and is not presented as a validated medical result.

## 4. Regression Scope

A separate regression dataset is **not** part of the initial implementation
scope.

- Regression may be added later, and only if the core classification work, the
  custom package, the notebooks, and the tests are complete and reliable.
- Deferring regression keeps the project focused on a single dataset and a
  single primary task, which makes the workflow easier to implement, document,
  test, and reproduce correctly.

## 5. Algorithm Scope

Algorithms are divided into three groups by how they are delivered.

### Custom package implementations

- Logistic regression implemented from scratch using gradient descent (the
  primary from-scratch classifier).
- A k-nearest neighbors classifier.
- Classification metrics: accuracy, precision, recall, F1, and confusion
  matrix.
- Preprocessing helpers: train/test split and standardization.

### scikit-learn comparison baselines

- `LogisticRegression`
- `KNeighborsClassifier`
- `DecisionTreeClassifier` or `RandomForestClassifier`

These baselines validate the custom implementations against trusted references
and provide context for the observed performance.

### Notebook-only demonstrations

- PCA
- K-means clustering
- Optionally, a simple SVM or neural-network baseline, only if time permits.

## 6. Notebook Scope

The proposed notebook set, each with one clear purpose:

- `01_data_exploration.ipynb` — Load the dataset, summarize features and class
  balance, and visualize distributions and correlations.
- `02_supervised_workflow.ipynb` — End-to-end supervised pipeline:
  preprocessing, training the custom classifier, and basic evaluation.
- `03_model_comparison.ipynb` — Compare the custom implementations against the
  scikit-learn baselines using shared metrics.
- `04_unsupervised_pca_clustering.ipynb` — PCA for dimensionality reduction and
  visualization, plus K-means clustering compared informally against the
  labels.
- `05_package_usage.ipynb` — A concise example showing how the `cmor438_ml`
  package is imported and used as a library.

## 7. Testing Implications

The test suite should eventually cover:

- **Metrics correctness.** Metric functions match hand-computed values on small,
  fixed examples.
- **Model fit/predict behavior.** Estimators expose `fit` / `predict` and behave
  as expected on simple inputs.
- **Shape validation.** Inputs and outputs have correct, validated shapes, and
  mismatched shapes raise clear errors.
- **Reproducibility.** Fixing `random_state` (where relevant) yields identical
  results across runs.
- **Expected behavior, not invented numbers.** Tests compare against simple,
  defensible expected behavior rather than against invented or assumed
  performance claims.

## 8. Decisions Deferred

The following remain open and are intentionally out of the initial scope:

- Whether to add a regression extension.
- Whether to add an SVM baseline.
- Whether to add a neural-network baseline.
- Whether to add advanced ensemble methods beyond the simple baselines.
