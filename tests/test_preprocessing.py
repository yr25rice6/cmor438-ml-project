import numpy as np
import pytest

from cmor438_ml.preprocessing import (
    StandardScalerScratch,
    add_intercept,
    train_test_split,
)
from cmor438_ml.validation import NotFittedError


# --- train_test_split -------------------------------------------------------

def _make_dataset(n_samples=10, n_features=3):
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    y = np.arange(n_samples)
    return X, y


def test_train_test_split_float_shapes():
    # 10 samples, test_size=0.25 -> ceil(2.5) = 3 test rows, 7 train rows.
    X, y = _make_dataset(n_samples=10, n_features=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    assert X_train.shape == (7, 3)
    assert X_test.shape == (3, 3)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_train_test_split_int_test_count():
    X, y = _make_dataset(n_samples=10, n_features=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=4, random_state=0
    )
    assert X_test.shape[0] == 4
    assert X_train.shape[0] == 6
    assert y_test.shape[0] == 4
    assert y_train.shape[0] == 6


def test_train_test_split_no_shuffle_preserves_order():
    X, y = _make_dataset(n_samples=6, n_features=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=2, shuffle=False
    )
    # First 2 rows are the test set, remaining rows are the train set, in order.
    assert np.array_equal(y_test, np.array([0, 1]))
    assert np.array_equal(y_train, np.array([2, 3, 4, 5]))
    assert np.array_equal(X_test, X[:2])
    assert np.array_equal(X_train, X[2:])


def test_train_test_split_same_random_state_is_deterministic():
    X, y = _make_dataset(n_samples=12, n_features=2)
    split_a = train_test_split(X, y, test_size=0.25, random_state=42)
    split_b = train_test_split(X, y, test_size=0.25, random_state=42)
    for first, second in zip(split_a, split_b):
        assert np.array_equal(first, second)


def test_train_test_split_different_random_state_usually_differs():
    X, y = _make_dataset(n_samples=50, n_features=2)
    _, _, y_train_a, _ = train_test_split(X, y, test_size=0.25, random_state=1)
    _, _, y_train_b, _ = train_test_split(X, y, test_size=0.25, random_state=2)
    # Distinct seeds should (with very high probability) yield different orders.
    assert not np.array_equal(y_train_a, y_train_b)


@pytest.mark.parametrize("bad_test_size", [0, 1.0, -0.5, 9])
def test_train_test_split_rejects_invalid_test_size(bad_test_size):
    # 9 is too large for 9 samples (leaves an empty train set); 0, 1.0, -0.5 are
    # out of the valid float/int ranges.
    X, y = _make_dataset(n_samples=9, n_features=2)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=bad_test_size)


def test_train_test_split_rejects_mismatched_lengths():
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0.25)


# --- StandardScalerScratch --------------------------------------------------

def test_standard_scaler_fit_stores_mean_and_scale():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    scaler = StandardScalerScratch()
    returned = scaler.fit(X)
    assert returned is scaler
    # Column means: [2, 20]; population std: column 0 -> sqrt(8/3), column 1 -> sqrt(200/3).
    assert np.allclose(scaler.mean_, np.array([2.0, 20.0]))
    assert np.allclose(
        scaler.scale_, np.array([np.sqrt(8.0 / 3.0), np.sqrt(200.0 / 3.0)])
    )


def test_standard_scaler_fit_transform_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(100, 4))
    transformed = StandardScalerScratch().fit_transform(X)
    assert np.allclose(transformed.mean(axis=0), np.zeros(4), atol=1e-12)
    # Population standard deviation of the standardized columns is 1.
    assert np.allclose(transformed.std(axis=0), np.ones(4), atol=1e-12)


def test_standard_scaler_transform_uses_training_statistics():
    X_fit = np.array([[0.0], [2.0], [4.0]])  # mean 2, population std sqrt(8/3)
    scaler = StandardScalerScratch().fit(X_fit)
    X_new = np.array([[2.0], [6.0]])
    expected = (X_new - 2.0) / np.sqrt(8.0 / 3.0)
    assert np.allclose(scaler.transform(X_new), expected)


def test_standard_scaler_constant_column_uses_scale_one():
    # Column 0 is constant -> scale 1.0 and transformed to zeros.
    X = np.array([[5.0, 0.0], [5.0, 2.0], [5.0, 4.0]])
    scaler = StandardScalerScratch()
    transformed = scaler.fit_transform(X)
    assert scaler.scale_[0] == 1.0
    assert np.allclose(transformed[:, 0], np.zeros(3))


def test_standard_scaler_transform_before_fit_raises():
    scaler = StandardScalerScratch()
    with pytest.raises(NotFittedError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_transform_rejects_wrong_column_count():
    scaler = StandardScalerScratch().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))


# --- add_intercept ----------------------------------------------------------

def test_add_intercept_adds_leading_ones_column():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = add_intercept(X)
    assert result.shape == (2, 3)
    assert np.array_equal(result[:, 0], np.ones(2))


def test_add_intercept_preserves_original_features():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = add_intercept(X)
    assert np.array_equal(result[:, 1:], X)


def test_add_intercept_does_not_mutate_input():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = X.copy()
    add_intercept(X)
    assert np.array_equal(X, original)


def test_add_intercept_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        add_intercept(np.array([1.0, 2.0, 3.0]))
