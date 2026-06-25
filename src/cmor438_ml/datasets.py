"""Dataset-loading helpers for the cmor438_ml package.

This module provides a thin, convenient wrapper around scikit-learn's built-in
Breast Cancer Wisconsin (Diagnostic) loader, which is the main dataset for the
CMOR 438 / INDE 577 final project. The data ships with scikit-learn, so loading
it requires no network access or downloaded files. Unlike the rest of the
package, this module depends on scikit-learn and pandas, which are project
dependencies used for data loading and the notebooks.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

__all__ = [
    "load_breast_cancer_data",
]


def load_breast_cancer_data(
    return_frame: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    The data is read from scikit-learn's bundled copy via
    :func:`sklearn.datasets.load_breast_cancer`; no external files are
    downloaded.

    Parameters
    ----------
    return_frame : bool, default=False
        If ``False``, return the data as plain NumPy arrays together with the
        feature and class names. If ``True``, return a single pandas
        ``DataFrame`` with the feature columns and a ``"target"`` column.

    Returns
    -------
    X : numpy.ndarray
        Two-dimensional feature matrix of shape ``(n_samples, n_features)``.
        Returned only when ``return_frame`` is ``False``.
    y : numpy.ndarray
        One-dimensional target vector of shape ``(n_samples,)``.
        Returned only when ``return_frame`` is ``False``.
    feature_names : numpy.ndarray
        One-dimensional array of feature names, one per column of ``X``.
        Returned only when ``return_frame`` is ``False``.
    target_names : numpy.ndarray
        One-dimensional array of the two class names. Returned only when
        ``return_frame`` is ``False``.
    frame : pandas.DataFrame
        A DataFrame with all feature columns plus a ``"target"`` column.
        Returned only when ``return_frame`` is ``True``.

    Raises
    ------
    ValueError
        If ``return_frame`` is not a ``bool``.
    """
    if not isinstance(return_frame, bool):
        raise ValueError(
            f"return_frame must be a bool, but got {type(return_frame).__name__}."
        )

    dataset = load_breast_cancer()

    if return_frame:
        frame = pd.DataFrame(
            data=np.asarray(dataset.data),
            columns=list(dataset.feature_names),
        )
        frame["target"] = np.asarray(dataset.target)
        return frame

    X = np.asarray(dataset.data)
    y = np.asarray(dataset.target)
    return X, y, dataset.feature_names, dataset.target_names
