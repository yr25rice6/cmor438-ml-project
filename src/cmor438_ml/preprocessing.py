"""Data-preparation helpers for the cmor438_ml package.

This module provides small, dependency-light utilities used to prepare data for
the classifiers in this project: a train/test splitter, a standardization
transformer, and an add-intercept helper for models that expect a bias column.
NumPy is the only dependency, and all inputs are validated with the shared
helpers in :mod:`cmor438_ml.validation`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from cmor438_ml.validation import check_array_2d, check_is_fitted, check_X_y

__all__ = [
    "train_test_split",
    "StandardScalerScratch",
    "add_intercept",
]


def train_test_split(
    X,
    y,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature and target arrays into train and test partitions.

    Parameters
    ----------
    X
        Array-like feature matrix; validated with :func:`check_X_y`.
    y
        Array-like target vector; validated with :func:`check_X_y`.
    test_size
        Size of the test partition. A float strictly between ``0`` and ``1`` is
        treated as a fraction of the samples (using ``ceil`` for the count); an
        int strictly between ``1`` and ``n_samples - 1`` (inclusive) is treated
        as the exact number of test samples.
    random_state
        Seed forwarded to :func:`numpy.random.default_rng` when ``shuffle`` is
        ``True``, for reproducible splits.
    shuffle
        Whether to shuffle the sample indices before splitting. When ``False``,
        the original order is preserved.

    Returns
    -------
    tuple of numpy.ndarray
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` are invalid (see :func:`check_X_y`) or if
        ``test_size`` is out of range for the number of samples.
    """
    X_array, y_array = check_X_y(X, y)
    n_samples = X_array.shape[0]

    n_test = _resolve_test_count(test_size, n_samples)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X_array[train_indices]
    X_test = X_array[test_indices]
    y_train = y_array[train_indices]
    y_test = y_array[test_indices]

    return X_train, X_test, y_train, y_test


def _resolve_test_count(test_size, n_samples: int) -> int:
    """Translate ``test_size`` into a valid test-sample count.

    A boolean is rejected outright (``bool`` is a subclass of ``int`` but is not
    a meaningful size). Floats are interpreted as fractions and ints as exact
    counts; both must leave the train and test partitions non-empty.
    """
    if isinstance(test_size, bool):
        raise ValueError("test_size must be a float or int, not a bool.")

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                "When test_size is a float it must be strictly between 0 and 1, "
                f"but got {test_size}."
            )
        n_test = math.ceil(n_samples * test_size)
    elif isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    else:
        raise ValueError(
            f"test_size must be a float or int, but got {type(test_size).__name__}."
        )

    if not 1 <= n_test <= n_samples - 1:
        raise ValueError(
            f"test_size resolves to {n_test} test sample(s), which does not "
            f"leave non-empty train and test partitions for {n_samples} sample(s)."
        )

    return n_test


class StandardScalerScratch:
    """Standardize features by removing the mean and scaling to unit variance.

    The scaler stores the per-feature mean (``mean_``) and standard deviation
    (``scale_``) learned during :meth:`fit`. Features with zero standard
    deviation are given a scale of ``1.0`` so that standardization maps their
    (constant) values to zeros without dividing by zero.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X) -> "StandardScalerScratch":
        """Compute and store the per-feature mean and standard deviation.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_array_2d`.

        Returns
        -------
        StandardScalerScratch
            The fitted scaler (``self``).
        """
        X_array = check_array_2d(X)

        self.mean_ = X_array.mean(axis=0)
        scale = X_array.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        self.n_features_in_ = X_array.shape[1]

        return self

    def transform(self, X) -> np.ndarray:
        """Standardize ``X`` using the stored mean and scale.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_array_2d`.

        Returns
        -------
        numpy.ndarray
            The standardized array ``(X - mean_) / scale_``.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        ValueError
            If ``X`` has a different number of columns than the fit data.
        """
        check_is_fitted(self, ["mean_", "scale_"])
        X_array = check_array_2d(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} feature(s), but this scaler was fitted "
                f"on {self.n_features_in_} feature(s)."
            )

        return (X_array - self.mean_) / self.scale_

    def fit_transform(self, X) -> np.ndarray:
        """Fit to ``X`` then return the standardized ``X``.

        Parameters
        ----------
        X
            Array-like feature matrix.

        Returns
        -------
        numpy.ndarray
            The standardized array.
        """
        return self.fit(X).transform(X)


def add_intercept(X) -> np.ndarray:
    """Prepend a leading column of ones to ``X`` for a bias term.

    Parameters
    ----------
    X
        Array-like feature matrix; validated with :func:`check_array_2d`.

    Returns
    -------
    numpy.ndarray
        A new array with shape ``(n_samples, n_features + 1)`` whose first column
        is all ones, followed by the original features. The input is not modified
        in place.

    Raises
    ------
    ValueError
        If ``X`` is not a valid two-dimensional array.
    """
    X_array = check_array_2d(X)

    ones = np.ones((X_array.shape[0], 1), dtype=X_array.dtype)
    return np.hstack((ones, X_array))
