"""Binary logistic regression for the cmor438_ml package.

This module implements a binary logistic regression classifier trained with
batch gradient descent on the binary cross-entropy loss, from scratch using only
NumPy. Inputs are validated with the shared helpers in
:mod:`cmor438_ml.validation`, and scoring reuses
:func:`cmor438_ml.metrics.accuracy_score`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cmor438_ml.metrics import accuracy_score
from cmor438_ml.validation import (
    check_array_2d,
    check_is_fitted,
    check_X_y,
    ensure_binary_labels,
)

__all__ = ["LogisticRegressionGD"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid of ``z``, computed in a numerically stable way.

    The linear scores are clipped before exponentiation so that very large
    magnitudes do not overflow :func:`numpy.exp`.
    """
    clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent.

    The classifier encodes the two observed labels as ``0`` and ``1`` (in sorted
    order), fits weights by minimizing the binary cross-entropy loss with full
    -batch gradient descent, and predicts the positive class (``classes_[1]``)
    when its estimated probability is at least ``0.5``.

    Parameters
    ----------
    learning_rate
        Step size applied to the gradient at each iteration. Must be a positive
        ``int`` or ``float`` (``bool`` is rejected).
    n_iterations
        Maximum number of gradient-descent iterations. Must be an integer greater
        than or equal to ``1`` (``bool`` is rejected).
    fit_intercept
        Whether to include a bias term in the model. Must be a ``bool``.
    tolerance
        Optional early-stopping threshold. When set, training stops once the
        absolute decrease in loss between two iterations is smaller than
        ``tolerance``. Must be ``None`` or a nonnegative ``int``/``float``
        (``bool`` is rejected).

    Attributes
    ----------
    coef_ : numpy.ndarray
        Feature coefficients with shape ``(n_features,)``, set by :meth:`fit`.
    intercept_ : float
        The bias term, set by :meth:`fit` (``0.0`` when ``fit_intercept`` is
        ``False``).
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    classes_ : numpy.ndarray
        Sorted unique labels seen during :meth:`fit`.
    n_iter_ : int
        Number of gradient-descent iterations actually performed.
    loss_history_ : list of float
        Binary cross-entropy loss after each performed iteration.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        tolerance: Optional[float] = None,
    ) -> None:
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"learning_rate must be a positive int or float, but got "
                f"{type(learning_rate).__name__}."
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be greater than 0, but got {learning_rate}."
            )

        if isinstance(n_iterations, bool) or not isinstance(
            n_iterations, (int, np.integer)
        ):
            raise ValueError(
                f"n_iterations must be an integer, but got "
                f"{type(n_iterations).__name__}."
            )
        if n_iterations < 1:
            raise ValueError(
                f"n_iterations must be greater than or equal to 1, but got "
                f"{n_iterations}."
            )

        if not isinstance(fit_intercept, bool):
            raise ValueError(
                f"fit_intercept must be a bool, but got "
                f"{type(fit_intercept).__name__}."
            )

        if tolerance is not None:
            if isinstance(tolerance, bool) or not isinstance(
                tolerance, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"tolerance must be None or a nonnegative int or float, but "
                    f"got {type(tolerance).__name__}."
                )
            if tolerance < 0:
                raise ValueError(
                    f"tolerance must be nonnegative, but got {tolerance}."
                )

        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.fit_intercept = fit_intercept
        self.tolerance = None if tolerance is None else float(tolerance)

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None
        self.loss_history_: Optional[List[float]] = None

    def fit(self, X, y) -> "LogisticRegressionGD":
        """Train the classifier on ``(X, y)`` via batch gradient descent.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_X_y`.
        y
            Array-like binary target vector; validated with :func:`check_X_y`
            and :func:`ensure_binary_labels`.

        Returns
        -------
        LogisticRegressionGD
            The fitted classifier (``self``).

        Raises
        ------
        ValueError
            If ``X`` and ``y`` are invalid (see :func:`check_X_y`) or if ``y``
            does not contain exactly two unique classes.
        """
        X_array, y_array = check_X_y(X, y)
        classes = ensure_binary_labels(y_array)

        # Encode labels as 0 for classes_[0] and 1 for classes_[1].
        y_encoded = (y_array == classes[1]).astype(float)

        n_samples, n_features = X_array.shape

        # Optionally prepend a bias column so the intercept is learned alongside
        # the feature weights.
        if self.fit_intercept:
            design = np.hstack((np.ones((n_samples, 1)), X_array))
        else:
            design = X_array

        weights = np.zeros(design.shape[1])
        loss_history: List[float] = []
        previous_loss: Optional[float] = None
        n_iter = 0

        for _ in range(self.n_iterations):
            scores = design @ weights
            probabilities = _sigmoid(scores)

            error = probabilities - y_encoded
            gradient = (design.T @ error) / n_samples
            weights = weights - self.learning_rate * gradient

            loss = self._binary_cross_entropy(y_encoded, _sigmoid(design @ weights))
            loss_history.append(loss)
            n_iter += 1

            if (
                self.tolerance is not None
                and previous_loss is not None
                and abs(previous_loss - loss) < self.tolerance
            ):
                break
            previous_loss = loss

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()

        self.n_features_in_ = n_features
        self.classes_ = classes
        self.n_iter_ = n_iter
        self.loss_history_ = loss_history

        return self

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        """Return the mean binary cross-entropy loss.

        Probabilities are clipped away from ``0`` and ``1`` so the logarithms
        stay finite.
        """
        eps = 1e-15
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        losses = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
        return float(np.mean(losses))

    def predict_proba(self, X) -> np.ndarray:
        """Return the predicted probability of ``classes_[1]`` for each row of ``X``.

        Parameters
        ----------
        X
            Array-like feature matrix; validated with :func:`check_array_2d`.

        Returns
        -------
        numpy.ndarray
            A one-dimensional array of probabilities in ``[0, 1]``, one per row.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted.
        ValueError
            If ``X`` is invalid or has a different number of columns than the
            training data.
        """
        check_is_fitted(self, ["coef_", "classes_"])
        X_array = check_array_2d(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} feature(s), but this classifier was "
                f"fitted on {self.n_features_in_} feature(s)."
            )

        scores = X_array @ self.coef_ + self.intercept_
        return _sigmoid(scores)

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X``.

        Parameters
        ----------
        X
            Array-like feature matrix.

        Returns
        -------
        numpy.ndarray
            A one-dimensional array with one predicted label per row of ``X``.
            ``classes_[1]`` is predicted when its probability is at least
            ``0.5``, otherwise ``classes_[0]``.
        """
        probabilities = self.predict_proba(X)
        positive = probabilities >= 0.5
        return np.where(positive, self.classes_[1], self.classes_[0])

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
