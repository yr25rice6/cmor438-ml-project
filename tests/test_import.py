"""Smoke tests for the cmor438_ml package-level public API.

These tests only confirm that the package imports, exposes a version, and
re-exports its public helpers, plus a tiny end-to-end usage example. The full
behavior of each helper is covered in test_validation.py, test_metrics.py, and
test_preprocessing.py.
"""

import numpy as np

import cmor438_ml


def test_package_imports():
    assert cmor438_ml is not None


def test_version_is_non_empty_string():
    assert hasattr(cmor438_ml, "__version__")
    assert isinstance(cmor438_ml.__version__, str)
    assert cmor438_ml.__version__ != ""


def test_validation_helpers_exported():
    for name in (
        "NotFittedError",
        "check_array_2d",
        "check_X_y",
        "check_is_fitted",
        "ensure_binary_labels",
    ):
        assert hasattr(cmor438_ml, name)


def test_metric_helpers_exported():
    for name in (
        "accuracy_score",
        "confusion_matrix",
        "precision_score",
        "recall_score",
        "f1_score",
    ):
        assert hasattr(cmor438_ml, name)


def test_preprocessing_helpers_exported():
    for name in ("train_test_split", "StandardScalerScratch", "add_intercept"):
        assert hasattr(cmor438_ml, name)


def test_evaluation_helpers_exported():
    for name in (
        "classification_report_dict",
        "evaluate_classifier",
        "compare_classifiers",
    ):
        assert hasattr(cmor438_ml, name)


def test_dataset_loader_exported_and_callable():
    assert hasattr(cmor438_ml, "load_breast_cancer_data")
    X, y, feature_names, target_names = cmor438_ml.load_breast_cancer_data()
    assert X.shape[0] == y.shape[0]
    assert len(feature_names) == X.shape[1]
    assert len(target_names) == 2


def test_package_level_usage_example():
    assert cmor438_ml.accuracy_score([0, 1], [0, 1]) == 1.0

    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    with_intercept = cmor438_ml.add_intercept(X)
    assert with_intercept.shape == (2, 3)
    assert np.all(with_intercept[:, 0] == 1.0)

    scaler = cmor438_ml.StandardScalerScratch()
    scaled = scaler.fit_transform(X)
    assert scaled.shape == X.shape
