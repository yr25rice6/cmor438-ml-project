"""Shared input-validation helpers for the cmor438_ml package.

These utilities coerce inputs to consistent NumPy arrays, check shapes, and
verify estimator fitted-state, so the models and preprocessing code can rely on
clean, well-formed inputs. NumPy is the only dependency.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "NotFittedError",
    "check_array_2d",
    "check_X_y",
    "check_is_fitted",
    "ensure_binary_labels",
]


class NotFittedError(RuntimeError):
    """Raised when an estimator is used before it has been fitted."""


def check_array_2d(X, name: str = "X", dtype=float) -> np.ndarray:
    """Convert ``X`` to a 2-D NumPy array and validate its shape.

    Parameters
    ----------
    X
        Array-like input to validate.
    name
        Name used in error messages.
    dtype
        Target dtype for the converted array.

    Returns
    -------
    numpy.ndarray
        The converted two-dimensional array.

    Raises
    ------
    ValueError
        If ``X`` cannot be converted, is not two-dimensional, or has no rows
        or no columns.
    """
    try:
        array = np.asarray(X, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} could not be converted to a numeric array: {exc}")

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional, but got {array.ndim} dimension(s) "
            f"with shape {array.shape}."
        )

    n_rows, n_cols = array.shape
    if n_rows < 1 or n_cols < 1:
        raise ValueError(
            f"{name} must have at least one row and one column, but got "
            f"shape {array.shape}."
        )

    return array


def check_X_y(X, y, dtype=float) -> Tuple[np.ndarray, np.ndarray]:
    """Validate ``X`` and ``y`` and check that their lengths align.

    Parameters
    ----------
    X
        Array-like feature matrix; validated with :func:`check_array_2d`.
    y
        Array-like target vector.
    dtype
        Target dtype for ``X``.

    Returns
    -------
    tuple of numpy.ndarray
        The converted ``(X, y)`` arrays.

    Raises
    ------
    ValueError
        If ``X`` is invalid, ``y`` is not one-dimensional, ``y`` is empty, or
        the number of samples in ``X`` and ``y`` differ.
    """
    X_array = check_array_2d(X, name="X", dtype=dtype)

    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional, but got {y_array.ndim} dimension(s) "
            f"with shape {y_array.shape}."
        )
    if y_array.shape[0] < 1:
        raise ValueError("y must contain at least one value, but it is empty.")

    if y_array.shape[0] != X_array.shape[0]:
        raise ValueError(
            f"X and y have mismatched sample counts: X has {X_array.shape[0]} "
            f"row(s) but y has {y_array.shape[0]} value(s)."
        )

    return X_array, y_array


def check_is_fitted(
    estimator, attributes: Union[str, Sequence[str]]
) -> None:
    """Verify that ``estimator`` has the given fitted attributes set.

    Parameters
    ----------
    estimator
        The estimator instance to check.
    attributes
        A single attribute name or a list/tuple of names that must exist on
        ``estimator`` and not be ``None``.

    Returns
    -------
    None
        If all required attributes are present and not ``None``.

    Raises
    ------
    NotFittedError
        If any required attribute is missing or ``None``.
    """
    if isinstance(attributes, str):
        names: Iterable[str] = [attributes]
    else:
        names = attributes

    missing = [
        name
        for name in names
        if not hasattr(estimator, name) or getattr(estimator, name) is None
    ]

    if missing:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet: "
            f"missing attribute(s) {missing}. Call 'fit' before using this method."
        )

    return None


def ensure_binary_labels(y) -> np.ndarray:
    """Validate that ``y`` is a one-dimensional vector with exactly two classes.

    Parameters
    ----------
    y
        Array-like target vector.

    Returns
    -------
    numpy.ndarray
        The two unique labels found in ``y``.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or does not contain exactly two unique
        labels.
    """
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional, but got {y_array.ndim} dimension(s) "
            f"with shape {y_array.shape}."
        )

    labels = np.unique(y_array)
    if labels.shape[0] != 2:
        raise ValueError(
            f"y must contain exactly two unique classes, but found "
            f"{labels.shape[0]}: {labels.tolist()}."
        )

    return labels
