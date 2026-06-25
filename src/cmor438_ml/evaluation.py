"""Model-evaluation utilities for the cmor438_ml package.

This module bundles the project's classification metrics into convenient
summaries: a per-prediction report dictionary, a fit-and-score helper for a
single estimator, and a comparison table across several estimators. It reuses
the metric implementations in :mod:`cmor438_ml.metrics` and the validation
helpers in :mod:`cmor438_ml.validation` rather than recomputing anything.
Building the comparison table requires pandas; the other helpers need only
NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmor438_ml.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from cmor438_ml.validation import check_X_y

__all__ = [
    "classification_report_dict",
    "evaluate_classifier",
    "compare_classifiers",
]


def classification_report_dict(y_true, y_pred, positive_label=1) -> dict:
    """Summarize binary-classification quality for one set of predictions.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.
    positive_label
        The label treated as the positive class for precision, recall, and F1.

    Returns
    -------
    dict
        A dictionary with keys ``"accuracy"``, ``"precision"``, ``"recall"``,
        ``"f1"`` (all floats), and ``"confusion_matrix"`` (the NumPy array
        returned by :func:`cmor438_ml.metrics.confusion_matrix`).

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` are not one-dimensional, are empty, or
        differ in length (see the underlying metric functions).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, positive_label=positive_label),
        "recall": recall_score(y_true, y_pred, positive_label=positive_label),
        "f1": f1_score(y_true, y_pred, positive_label=positive_label),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def _check_estimator(estimator) -> None:
    """Raise ``ValueError`` unless ``estimator`` has callable fit and predict."""
    for method in ("fit", "predict"):
        attribute = getattr(estimator, method, None)
        if not callable(attribute):
            raise ValueError(
                f"estimator must expose a callable '{method}' method, but "
                f"{type(estimator).__name__} does not."
            )


def evaluate_classifier(
    estimator, X_train, X_test, y_train, y_test, positive_label=1
) -> dict:
    """Fit an estimator and report its train and test performance.

    Parameters
    ----------
    estimator
        An object exposing callable ``fit(X, y)`` and ``predict(X)`` methods.
    X_train, y_train
        Array-like training features and labels; validated with
        :func:`cmor438_ml.validation.check_X_y`.
    X_test, y_test
        Array-like test features and labels; validated with
        :func:`cmor438_ml.validation.check_X_y`.
    positive_label
        The label treated as the positive class for precision, recall, and F1.

    Returns
    -------
    dict
        A dictionary with keys ``"train_accuracy"``, ``"test_accuracy"``,
        ``"test_precision"``, ``"test_recall"``, ``"test_f1"``, and
        ``"test_confusion_matrix"``.

    Raises
    ------
    ValueError
        If ``estimator`` lacks a callable ``fit`` or ``predict`` method, or if
        the train/test arrays are invalid (see
        :func:`cmor438_ml.validation.check_X_y`).
    """
    _check_estimator(estimator)
    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)

    estimator.fit(X_train, y_train)
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)

    train_report = classification_report_dict(
        y_train, train_pred, positive_label=positive_label
    )
    test_report = classification_report_dict(
        y_test, test_pred, positive_label=positive_label
    )

    return {
        "train_accuracy": train_report["accuracy"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["precision"],
        "test_recall": test_report["recall"],
        "test_f1": test_report["f1"],
        "test_confusion_matrix": test_report["confusion_matrix"],
    }


def compare_classifiers(
    classifiers, X_train, X_test, y_train, y_test, positive_label=1
) -> pd.DataFrame:
    """Evaluate several estimators and tabulate their scores side by side.

    Parameters
    ----------
    classifiers
        Dictionary mapping model names to estimator objects, each exposing
        callable ``fit`` and ``predict`` methods. Insertion order is preserved
        in the output rows.
    X_train, y_train
        Array-like training features and labels.
    X_test, y_test
        Array-like test features and labels.
    positive_label
        The label treated as the positive class for precision, recall, and F1.

    Returns
    -------
    pandas.DataFrame
        One row per model, with columns ``"model"``, ``"train_accuracy"``,
        ``"test_accuracy"``, ``"test_precision"``, ``"test_recall"``, and
        ``"test_f1"``. Confusion matrices are not included.

    Raises
    ------
    ValueError
        If ``classifiers`` is not a dictionary, is empty, or contains an
        estimator without a callable ``fit`` or ``predict`` method.
    """
    if not isinstance(classifiers, dict):
        raise ValueError(
            f"classifiers must be a dict, but got {type(classifiers).__name__}."
        )
    if not classifiers:
        raise ValueError("classifiers must not be empty.")

    rows = []
    for name, estimator in classifiers.items():
        result = evaluate_classifier(
            estimator, X_train, X_test, y_train, y_test, positive_label=positive_label
        )
        rows.append(
            {
                "model": name,
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "test_precision": result["test_precision"],
                "test_recall": result["test_recall"],
                "test_f1": result["test_f1"],
            }
        )

    columns = [
        "model",
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    return pd.DataFrame(rows, columns=columns)
