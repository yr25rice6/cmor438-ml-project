"""Tests for the cross-validation utilities in cmor438_ml.model_selection."""

import numpy as np
import pandas as pd
import pytest

from cmor438_ml.model_selection import cross_validate_classifier, k_fold_split
from cmor438_ml.models import KNNClassifier


def _simple_dataset(n_samples=12, n_features=3):
    """Build a small, deterministic two-class dataset."""
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    y = np.array([0, 1] * (n_samples // 2))
    return X, y


def _separable_dataset():
    """Build a tiny linearly separable two-class dataset for KNN."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.3],
            [0.3, 0.2],
            [5.0, 5.0],
            [5.2, 5.1],
            [5.1, 5.3],
            [5.3, 5.2],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


# --- k_fold_split -----------------------------------------------------------


def test_k_fold_split_returns_requested_number_of_folds():
    X, y = _simple_dataset()
    folds = k_fold_split(X, y, n_splits=4)
    assert isinstance(folds, list)
    assert len(folds) == 4


def test_each_fold_has_valid_shapes():
    X, y = _simple_dataset(n_samples=12, n_features=3)
    folds = k_fold_split(X, y, n_splits=4, shuffle=False)
    for X_train, X_test, y_train, y_test in folds:
        assert X_train.shape[1] == 3
        assert X_test.shape[1] == 3
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]


def test_every_sample_in_exactly_one_test_fold():
    X, y = _simple_dataset(n_samples=12)
    folds = k_fold_split(X, y, n_splits=4, shuffle=False)

    seen = []
    for _, X_test, _, _ in folds:
        seen.extend(X_test[:, 0].tolist())

    # With shuffle=False the first column values identify samples uniquely.
    assert sorted(seen) == sorted(X[:, 0].tolist())
    assert len(seen) == len(set(seen)) == X.shape[0]


def test_fold_sizes_are_balanced_when_not_divisible():
    X, y = _simple_dataset(n_samples=10)
    folds = k_fold_split(X, y, n_splits=3, shuffle=False)
    test_sizes = [X_test.shape[0] for _, X_test, _, _ in folds]
    # 10 samples into 3 folds -> sizes 4, 3, 3.
    assert sorted(test_sizes) == [3, 3, 4]
    assert max(test_sizes) - min(test_sizes) <= 1
    assert sum(test_sizes) == X.shape[0]


def test_shuffle_false_preserves_ordered_test_blocks():
    X, y = _simple_dataset(n_samples=12)
    folds = k_fold_split(X, y, n_splits=4, shuffle=False)

    expected_start = 0
    for _, X_test, _, _ in folds:
        block = X_test[:, 0].tolist()
        expected = X[expected_start : expected_start + len(block), 0].tolist()
        assert block == expected
        expected_start += len(block)


def test_same_random_state_gives_identical_folds():
    X, y = _simple_dataset()
    folds_a = k_fold_split(X, y, n_splits=4, shuffle=True, random_state=42)
    folds_b = k_fold_split(X, y, n_splits=4, shuffle=True, random_state=42)

    for (_, Xa, _, ya), (_, Xb, _, yb) in zip(folds_a, folds_b):
        assert np.array_equal(Xa, Xb)
        assert np.array_equal(ya, yb)


def test_different_random_state_usually_differs():
    X, y = _simple_dataset()
    folds_a = k_fold_split(X, y, n_splits=4, shuffle=True, random_state=0)
    folds_b = k_fold_split(X, y, n_splits=4, shuffle=True, random_state=99)

    test_blocks_a = [X_test[:, 0].tolist() for _, X_test, _, _ in folds_a]
    test_blocks_b = [X_test[:, 0].tolist() for _, X_test, _, _ in folds_b]
    assert test_blocks_a != test_blocks_b


@pytest.mark.parametrize("bad_n_splits", [1, 0, -1, 2.5, True])
def test_invalid_n_splits_raises(bad_n_splits):
    X, y = _simple_dataset()
    with pytest.raises(ValueError):
        k_fold_split(X, y, n_splits=bad_n_splits)


def test_n_splits_greater_than_n_samples_raises():
    X, y = _simple_dataset(n_samples=6)
    with pytest.raises(ValueError):
        k_fold_split(X, y, n_splits=7)


# --- cross_validate_classifier ---------------------------------------------


def test_cross_validate_returns_dataframe():
    X, y = _separable_dataset()
    result = cross_validate_classifier(
        lambda: KNNClassifier(n_neighbors=1), X, y, n_splits=4, shuffle=False
    )
    assert isinstance(result, pd.DataFrame)


def test_cross_validate_one_row_per_fold():
    X, y = _separable_dataset()
    result = cross_validate_classifier(
        lambda: KNNClassifier(n_neighbors=1), X, y, n_splits=4, shuffle=False
    )
    assert len(result) == 4


def test_cross_validate_has_required_columns():
    X, y = _separable_dataset()
    result = cross_validate_classifier(
        lambda: KNNClassifier(n_neighbors=1), X, y, n_splits=4, shuffle=False
    )
    expected = [
        "fold",
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    assert list(result.columns) == expected
    assert "test_confusion_matrix" not in result.columns


def test_cross_validate_fold_column_starts_at_one():
    X, y = _separable_dataset()
    result = cross_validate_classifier(
        lambda: KNNClassifier(n_neighbors=1), X, y, n_splits=4, shuffle=False
    )
    assert result["fold"].tolist() == [1, 2, 3, 4]


def test_cross_validate_works_with_knn_on_separable_data():
    X, y = _separable_dataset()
    result = cross_validate_classifier(
        lambda: KNNClassifier(n_neighbors=1),
        X,
        y,
        n_splits=4,
        shuffle=True,
        random_state=0,
    )
    # Each test fold mixes both clusters, and 1-NN separates them perfectly.
    assert (result["test_accuracy"] == 1.0).all()


def test_cross_validate_non_callable_factory_raises():
    X, y = _separable_dataset()
    not_callable = KNNClassifier(n_neighbors=1)
    with pytest.raises(ValueError):
        cross_validate_classifier(not_callable, X, y, n_splits=4)


def test_cross_validate_factory_without_fit_or_predict_raises():
    X, y = _separable_dataset()

    class NotAnEstimator:
        pass

    with pytest.raises(ValueError):
        cross_validate_classifier(lambda: NotAnEstimator(), X, y, n_splits=4)
