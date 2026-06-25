"""K-nearest neighbors classifier for the cmor438_ml package.

This module implements a simple, readable k-nearest neighbors classifier from
scratch using only NumPy. Inputs are validated with the shared helpers in
:mod:`cmor438_ml.validation`, and scoring reuses :func:`cmor438_ml.metrics.accuracy_score`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cmor438_ml.metrics import accuracy_score
from cmor438_ml.validation import check_array_2d, check_is_fitted, check_X_y

__all__ = ["KNNClassifier"]


class KNNClassifier:
    """Classify samples by majority vote among their nearest neighbors.

    The classifier memorizes the training data during :meth:`fit` and, for each
    query sample, predicts the most common label among the ``n_neighbors``
    closest training samples under Euclidean distance.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to consult for each prediction. Must be
        an integer greater than or equal to ``1``.

    Attributes
    ----------
    X_train_ : numpy.ndarray
        Copy of the training feature matrix, set by :meth:`fit`.
    y_train_ : numpy.ndarray
        Copy of the training labels, set by :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    classes_ : numpy.ndarray
        Sorted unique labels seen during :meth:`fit`.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)):
            raise ValueError(
                f"n_neighbors must be an integer, but got "
                f"{type(n_neighbors).__name__}."
            )
        if n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be greater than or equal to 1, but got "
                f"{n_neighbors}."
            )

        self.n_neighbors = int(n_neighbors)
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> "KNNClassifier":
        """Store the training data after validating it.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_X_y`.
        y
            Array-like target vector; validated with :func:`check_X_y`.

        Returns
        -------
        KNNClassifier
            The fitted classifier (``self``).

        Raises
        ------
        ValueError
            If ``X`` and ``y`` are invalid (see :func:`check_X_y`) or if
            ``n_neighbors`` exceeds the number of training samples.
        """
        X_array, y_array = check_X_y(X, y)

        if self.n_neighbors > X_array.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} is larger than the number of "
                f"training samples ({X_array.shape[0]})."
            )

        self.X_train_ = X_array.copy()
        self.y_train_ = y_array.copy()
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.unique(y_array)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict labels for ``X`` by majority vote among nearest neighbors.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_array_2d`.

        Returns
        -------
        numpy.ndarray
            A one-dimensional array with one predicted label per row of ``X``.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted.
        ValueError
            If ``X`` is invalid or has a different number of columns than the
            training data.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        X_array = check_array_2d(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} feature(s), but this classifier was "
                f"fitted on {self.n_features_in_} feature(s)."
            )

        # Pairwise Euclidean distances: distances[i, j] is the distance from
        # query row i to training row j. Squared distances suffice for ranking.
        differences = X_array[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        squared_distances = np.sum(differences ** 2, axis=2)

        predictions = np.empty(X_array.shape[0], dtype=self.y_train_.dtype)
        for i in range(X_array.shape[0]):
            neighbor_indices = np.argsort(squared_distances[i], kind="stable")[
                : self.n_neighbors
            ]
            neighbor_labels = self.y_train_[neighbor_indices]
            predictions[i] = self._majority_vote(neighbor_labels)

        return predictions

    def _majority_vote(self, neighbor_labels: np.ndarray):
        """Return the most frequent label, breaking ties by sorted label order.

        ``np.unique`` returns labels in sorted order, so the first label whose
        count equals the maximum count is the smallest one under that order,
        giving deterministic tie-breaking.
        """
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def score(self, X, y) -> float:
        """Return the accuracy of the classifier on ``(X, y)``.

        Parameters
        ----------
        X
            Array-like feature matrix.
        y
            Array-like true labels.

        Returns
        -------
        float
            The fraction of samples in ``X`` whose predicted label matches ``y``.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
