"""Tests for the cmor438_ml.evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from cmor438_ml.evaluation import (
    classification_report_dict,
    compare_classifiers,
    evaluate_classifier,
)
from cmor438_ml.models import KNNClassifier


# --- a small, easily separable two-class dataset ----------------------------

def _separable_dataset():
    """Return train/test splits for a trivially separable binary problem."""
    X_train = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [5.0, 5.0], [5.2, 4.9], [4.8, 5.1]]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.1, 0.0], [4.9, 5.0]])
    y_test = np.array([0, 1])
    return X_train, X_test, y_train, y_test


class _NoFitPredict:
    """Estimator stand-in that lacks both fit and predict methods."""


# --- classification_report_dict ---------------------------------------------

def test_report_returns_all_expected_keys():
    report = classification_report_dict([0, 1], [0, 1])
    assert set(report) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}


def test_report_matches_hand_computed_values():
    # y_true / y_pred chosen so the four counts are known by hand:
    #   TP=2 (positions 0, 1), FN=1 (position 2),
    #   FP=1 (position 3), TN=2 (positions 4, 5).
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]

    report = classification_report_dict(y_true, y_pred, positive_label=1)

    # accuracy = (TP + TN) / total = (2 + 2) / 6
    assert report["accuracy"] == pytest.approx(4.0 / 6.0)
    # precision = TP / (TP + FP) = 2 / 3
    assert report["precision"] == pytest.approx(2.0 / 3.0)
    # recall = TP / (TP + FN) = 2 / 3
    assert report["recall"] == pytest.approx(2.0 / 3.0)
    # f1 = harmonic mean of precision and recall = 2/3
    assert report["f1"] == pytest.approx(2.0 / 3.0)


def test_report_confusion_matrix_has_expected_values():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]

    matrix = classification_report_dict(y_true, y_pred)["confusion_matrix"]

    assert isinstance(matrix, np.ndarray)
    # Labels are sorted ascending: row/col 0 -> label 0, row/col 1 -> label 1.
    #   true 0: predicted 0 twice, predicted 1 once  -> [2, 1]
    #   true 1: predicted 0 once,  predicted 1 twice -> [1, 2]
    expected = np.array([[2, 1], [1, 2]])
    assert np.array_equal(matrix, expected)


# --- evaluate_classifier ----------------------------------------------------

def test_evaluate_classifier_returns_required_keys():
    X_train, X_test, y_train, y_test = _separable_dataset()
    result = evaluate_classifier(
        KNNClassifier(n_neighbors=1), X_train, X_test, y_train, y_test
    )
    assert set(result) == {
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_confusion_matrix",
    }


def test_evaluate_classifier_high_accuracy_on_separable_data():
    X_train, X_test, y_train, y_test = _separable_dataset()
    result = evaluate_classifier(
        KNNClassifier(n_neighbors=1), X_train, X_test, y_train, y_test
    )
    assert result["test_accuracy"] == pytest.approx(1.0)


def test_evaluate_classifier_rejects_object_without_fit_or_predict():
    X_train, X_test, y_train, y_test = _separable_dataset()
    with pytest.raises(ValueError):
        evaluate_classifier(_NoFitPredict(), X_train, X_test, y_train, y_test)


# --- compare_classifiers ----------------------------------------------------

def test_compare_classifiers_returns_dataframe():
    X_train, X_test, y_train, y_test = _separable_dataset()
    table = compare_classifiers(
        {"knn": KNNClassifier(n_neighbors=1)}, X_train, X_test, y_train, y_test
    )
    assert isinstance(table, pd.DataFrame)


def test_compare_classifiers_one_row_per_model():
    X_train, X_test, y_train, y_test = _separable_dataset()
    classifiers = {
        "knn1": KNNClassifier(n_neighbors=1),
        "knn3": KNNClassifier(n_neighbors=3),
    }
    table = compare_classifiers(classifiers, X_train, X_test, y_train, y_test)
    assert table.shape[0] == len(classifiers)


def test_compare_classifiers_has_required_columns():
    X_train, X_test, y_train, y_test = _separable_dataset()
    table = compare_classifiers(
        {"knn": KNNClassifier(n_neighbors=1)}, X_train, X_test, y_train, y_test
    )
    assert list(table.columns) == [
        "model",
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]


def test_compare_classifiers_preserves_model_names():
    X_train, X_test, y_train, y_test = _separable_dataset()
    classifiers = {
        "knn1": KNNClassifier(n_neighbors=1),
        "knn3": KNNClassifier(n_neighbors=3),
    }
    table = compare_classifiers(classifiers, X_train, X_test, y_train, y_test)
    assert list(table["model"]) == ["knn1", "knn3"]


def test_compare_classifiers_rejects_empty_dict():
    X_train, X_test, y_train, y_test = _separable_dataset()
    with pytest.raises(ValueError):
        compare_classifiers({}, X_train, X_test, y_train, y_test)


def test_compare_classifiers_rejects_non_dict():
    X_train, X_test, y_train, y_test = _separable_dataset()
    with pytest.raises(ValueError):
        compare_classifiers(
            [KNNClassifier(n_neighbors=1)], X_train, X_test, y_train, y_test
        )


def test_compare_classifiers_rejects_model_without_fit_or_predict():
    X_train, X_test, y_train, y_test = _separable_dataset()
    with pytest.raises(ValueError):
        compare_classifiers(
            {"bad": _NoFitPredict()}, X_train, X_test, y_train, y_test
        )
