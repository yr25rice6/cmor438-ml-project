"""Cross-validation utilities for the cmor438_ml package.

This module provides k-fold splitting and a cross-validation driver for binary
classifiers. Inputs are validated with the shared helpers in
:mod:`cmor438_ml.validation`, and per-fold scoring reuses
:func:`cmor438_ml.evaluation.evaluate_classifier` rather than recomputing any
metric formulas. Splitting needs only NumPy; the cross-validation summary
returns a pandas DataFrame.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from cmor438_ml.evaluation import evaluate_classifier
from cmor438_ml.validation import check_X_y

__all__ = [
    "k_fold_split",
    "cross_validate_classifier",
]


def k_fold_split(
    X, y, n_splits: int = 5, shuffle: bool = True, random_state=None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Partition samples into ``n_splits`` train/test folds.

    Parameters
    ----------
    X
        Array-like feature matrix; validated with
        :func:`cmor438_ml.validation.check_X_y`.
    y
        Array-like target vector; validated alongside ``X``.
    n_splits
        Number of folds. Must be an integer with ``2 <= n_splits <= n_samples``.
    shuffle
        Whether to shuffle the sample indices before splitting. When ``False``,
        the original sample order is preserved and each test fold is a
        contiguous block.
    random_state
        Seed forwarded to :func:`numpy.random.default_rng` when ``shuffle`` is
        ``True``, for reproducible folds.

    Returns
    -------
    list of tuple of numpy.ndarray
        A list of length ``n_splits``. Each element is a tuple
        ``(X_train, X_test, y_train, y_test)``. Every sample appears in exactly
        one test fold, and fold sizes differ by at most one sample.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` are invalid (see
        :func:`cmor438_ml.validation.check_X_y`), or if ``n_splits`` is not an
        integer, is a bool, is less than ``2``, or exceeds ``n_samples``.
    """
    X_array, y_array = check_X_y(X, y)
    n_samples = X_array.shape[0]

    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError(
            f"n_splits must be an integer, but got {type(n_splits).__name__}."
        )
    if n_splits < 2:
        raise ValueError(
            f"n_splits must be greater than or equal to 2, but got {n_splits}."
        )
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits={n_splits} is larger than the number of samples "
            f"({n_samples})."
        )

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Distribute samples so the first (n_samples % n_splits) folds get one extra
    # sample, keeping fold sizes as balanced as possible.
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))

        folds.append(
            (
                X_array[train_indices],
                X_array[test_indices],
                y_array[train_indices],
                y_array[test_indices],
            )
        )
        start = stop

    return folds


def _check_estimator_factory(estimator_factory) -> None:
    """Raise ``ValueError`` unless ``estimator_factory`` yields a valid estimator.

    The factory must be callable and return an object exposing callable ``fit``
    and ``predict`` methods.
    """
    if not callable(estimator_factory):
        raise ValueError(
            f"estimator_factory must be callable, but got "
            f"{type(estimator_factory).__name__}."
        )

    estimator = estimator_factory()
    for method in ("fit", "predict"):
        if not callable(getattr(estimator, method, None)):
            raise ValueError(
                f"estimator_factory must return an estimator with a callable "
                f"'{method}' method, but {type(estimator).__name__} does not."
            )


def cross_validate_classifier(
    estimator_factory,
    X,
    y,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state=None,
    positive_label=1,
) -> pd.DataFrame:
    """Cross-validate a classifier and tabulate per-fold scores.

    For each fold produced by :func:`k_fold_split`, a fresh estimator is built
    with ``estimator_factory()`` and scored with
    :func:`cmor438_ml.evaluation.evaluate_classifier`.

    Parameters
    ----------
    estimator_factory
        Zero-argument callable returning a fresh estimator object, each
        exposing callable ``fit`` and ``predict`` methods.
    X
        Array-like feature matrix; validated with
        :func:`cmor438_ml.validation.check_X_y`.
    y
        Array-like target vector; validated alongside ``X``.
    n_splits
        Number of folds passed to :func:`k_fold_split`.
    shuffle
        Whether to shuffle indices before splitting (see :func:`k_fold_split`).
    random_state
        Seed for reproducible folds (see :func:`k_fold_split`).
    positive_label
        The label treated as the positive class for precision, recall, and F1.

    Returns
    -------
    pandas.DataFrame
        One row per fold, with columns ``"fold"``, ``"train_accuracy"``,
        ``"test_accuracy"``, ``"test_precision"``, ``"test_recall"``, and
        ``"test_f1"``. Fold numbering starts at ``1``; confusion matrices are
        not included.

    Raises
    ------
    ValueError
        If ``estimator_factory`` is not callable or does not return an estimator
        with callable ``fit`` and ``predict`` methods, if ``X`` and ``y`` are
        invalid, or if ``n_splits`` is out of range (see :func:`k_fold_split`).
    """
    _check_estimator_factory(estimator_factory)

    folds = k_fold_split(
        X, y, n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    rows = []
    for fold_number, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        result = evaluate_classifier(
            estimator_factory(),
            X_train,
            X_test,
            y_train,
            y_test,
            positive_label=positive_label,
        )
        rows.append(
            {
                "fold": fold_number,
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "test_precision": result["test_precision"],
                "test_recall": result["test_recall"],
                "test_f1": result["test_f1"],
            }
        )

    columns = [
        "fold",
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    return pd.DataFrame(rows, columns=columns)
