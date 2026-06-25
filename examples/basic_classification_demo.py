"""Basic classification demo for the cmor438_ml package.

A concise, runnable example that ties the package together on the Breast Cancer
Wisconsin (Diagnostic) dataset. It loads the data, splits it into train and test
partitions, standardizes the features, trains the two from-scratch classifiers
(``KNNClassifier`` and ``LogisticRegressionGD``), and prints a short summary of
their test-set metrics. A ROC/AUC summary is included for the logistic
regression model.

Run it from the repository root with:

    .\\.venv\\Scripts\\python.exe examples/basic_classification_demo.py
"""

from __future__ import annotations

from cmor438_ml import (
    StandardScalerScratch,
    accuracy_score,
    auc_score,
    f1_score,
    load_breast_cancer_data,
    precision_score,
    recall_score,
    roc_curve_binary,
    train_test_split,
)
from cmor438_ml.models import KNNClassifier, LogisticRegressionGD

RANDOM_STATE = 42


def _evaluate(y_true, y_pred) -> dict:
    """Return the core classification metrics for a set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def _print_metrics(name: str, metrics: dict) -> None:
    """Print a model's metrics as a small aligned block."""
    print(f"{name}")
    for label in ("accuracy", "precision", "recall", "f1"):
        print(f"  {label:<10} {metrics[label]:.4f}")


def main() -> None:
    # Load the dataset as plain NumPy arrays. The default labels encode malignant
    # as 0 and benign as 1, so benign (label 1) is the positive class below.
    X, y, _feature_names, target_names = load_breast_cancer_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Fit the scaler on the training set only, then reuse it to transform both
    # partitions so no test information leaks into the fitted statistics.
    scaler = StandardScalerScratch()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Breast Cancer Wisconsin classification demo")
    print(f"  classes:        {[str(name) for name in target_names]}")
    print(f"  train samples:  {X_train.shape[0]}")
    print(f"  test samples:   {X_test.shape[0]}")
    print(f"  features:       {X_train.shape[1]}")
    print()

    # Train and evaluate the k-nearest-neighbors classifier.
    knn = KNNClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_metrics = _evaluate(y_test, knn.predict(X_test_scaled))
    _print_metrics("KNNClassifier (k=5)", knn_metrics)
    print()

    # Train and evaluate the logistic regression classifier.
    logreg = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_metrics = _evaluate(y_test, logreg.predict(X_test_scaled))
    _print_metrics("LogisticRegressionGD", logreg_metrics)

    # Use the predicted probabilities to summarize ranking quality across
    # thresholds via the ROC curve and its area.
    probabilities = logreg.predict_proba(X_test_scaled)
    fpr, tpr, _thresholds = roc_curve_binary(y_test, probabilities, positive_label=1)
    print(f"  {'roc auc':<10} {auc_score(fpr, tpr):.4f}")


if __name__ == "__main__":
    main()
