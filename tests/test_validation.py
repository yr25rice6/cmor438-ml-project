import numpy as np
import pytest

from cmor438_ml.validation import (
    NotFittedError,
    check_array_2d,
    check_X_y,
    check_is_fitted,
    ensure_binary_labels,
)


# --- check_array_2d ---------------------------------------------------------

def test_check_array_2d_accepts_valid_list_of_lists():
    result = check_array_2d([[1, 2], [3, 4]])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_check_array_2d_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        check_array_2d([1, 2, 3])


def test_check_array_2d_rejects_empty_input():
    with pytest.raises(ValueError):
        check_array_2d([])


# --- check_X_y --------------------------------------------------------------

def test_check_X_y_accepts_valid_inputs():
    X, y = check_X_y([[1, 2], [3, 4], [5, 6]], [0, 1, 0])
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_check_X_y_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError):
        check_X_y([[1, 2], [3, 4]], [0, 1, 0])


def test_check_X_y_rejects_two_dimensional_y():
    with pytest.raises(ValueError):
        check_X_y([[1, 2], [3, 4]], [[0], [1]])


# --- check_is_fitted --------------------------------------------------------

class _Estimator:
    pass


def test_check_is_fitted_passes_when_attributes_present():
    est = _Estimator()
    est.coef_ = np.array([1.0, 2.0])
    est.intercept_ = 0.5
    assert check_is_fitted(est, ["coef_", "intercept_"]) is None


def test_check_is_fitted_raises_when_attribute_missing():
    est = _Estimator()
    with pytest.raises(NotFittedError):
        check_is_fitted(est, "coef_")


def test_check_is_fitted_raises_when_attribute_is_none():
    est = _Estimator()
    est.coef_ = None
    with pytest.raises(NotFittedError):
        check_is_fitted(est, "coef_")


# --- ensure_binary_labels ---------------------------------------------------

def test_ensure_binary_labels_returns_two_labels():
    labels = ensure_binary_labels([0, 1, 1, 0, 1])
    assert len(labels) == 2
    assert set(labels.tolist()) == {0, 1}


def test_ensure_binary_labels_rejects_one_class():
    with pytest.raises(ValueError):
        ensure_binary_labels([1, 1, 1])


def test_ensure_binary_labels_rejects_three_classes():
    with pytest.raises(ValueError):
        ensure_binary_labels([0, 1, 2])
