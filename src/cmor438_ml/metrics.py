"""Classification metrics for the cmor438_ml package.

This module implements common classification metrics from scratch using only
NumPy. The functions target binary classification workflows but the accuracy
and confusion-matrix helpers also work for multiclass inputs. NumPy is the only
dependency.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
]


def _check_classification_targets(y_true, y_pred):
    """Convert targets to 1-D NumPy arrays and validate their shapes.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.

    Returns
    -------
    tuple of numpy.ndarray
        The converted ``(y_true, y_pred)`` arrays.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, if either input is empty, or if
        the two inputs have different lengths.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError(
            "y_true and y_pred must both be one-dimensional, but got shapes "
            f"{y_true_array.shape} and {y_pred_array.shape}."
        )

    if y_true_array.shape[0] == 0 or y_pred_array.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length, but got "
            f"{y_true_array.shape[0]} and {y_pred_array.shape[0]}."
        )

    return y_true_array, y_pred_array


def accuracy_score(y_true, y_pred) -> float:
    """Return the fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.

    Returns
    -------
    float
        The fraction of matching labels, in the range ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or differ in length.
    """
    y_true_array, y_pred_array = _check_classification_targets(y_true, y_pred)
    return float(np.mean(y_true_array == y_pred_array))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Compute a confusion matrix for the given labels.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.
    labels
        Optional sequence giving the label order for the matrix rows and
        columns. If ``None``, the sorted unique labels found in ``y_true`` and
        ``y_pred`` combined are used.

    Returns
    -------
    numpy.ndarray
        A 2-D integer array of shape ``(n_labels, n_labels)`` where entry
        ``(i, j)`` counts samples whose true label is ``labels[i]`` and whose
        predicted label is ``labels[j]``.

    Raises
    ------
    ValueError
        If the inputs are invalid (see :func:`_check_classification_targets`),
        if ``labels`` is empty, or if ``labels`` omits a label that appears in
        ``y_true`` or ``y_pred``.
    """
    y_true_array, y_pred_array = _check_classification_targets(y_true, y_pred)

    if labels is None:
        labels_array = np.unique(
            np.concatenate((y_true_array, y_pred_array))
        )
    else:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1 or labels_array.shape[0] == 0:
            raise ValueError("labels must be a non-empty one-dimensional sequence.")

        observed = np.unique(np.concatenate((y_true_array, y_pred_array)))
        allowed = set(labels_array.tolist())
        missing = [value for value in observed.tolist() if value not in allowed]
        if missing:
            raise ValueError(
                f"labels is missing observed class(es) {missing}."
            )

    label_to_index = {value: index for index, value in enumerate(labels_array.tolist())}
    n_labels = labels_array.shape[0]
    matrix = np.zeros((n_labels, n_labels), dtype=int)

    for true_value, pred_value in zip(y_true_array.tolist(), y_pred_array.tolist()):
        matrix[label_to_index[true_value], label_to_index[pred_value]] += 1

    return matrix


def _binary_counts(y_true_array, y_pred_array, positive_label):
    """Return ``(tp, fp, fn)`` counts for the given positive label."""
    actual_positive = y_true_array == positive_label
    predicted_positive = y_pred_array == positive_label

    true_positive = int(np.sum(actual_positive & predicted_positive))
    false_positive = int(np.sum(~actual_positive & predicted_positive))
    false_negative = int(np.sum(actual_positive & ~predicted_positive))

    return true_positive, false_positive, false_negative


def precision_score(y_true, y_pred, positive_label=1, zero_division=0.0) -> float:
    """Compute binary precision: ``TP / (TP + FP)``.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.
    positive_label
        The label treated as the positive class.
    zero_division
        Value returned when there are no predicted positives (zero denominator).

    Returns
    -------
    float
        The precision, or ``zero_division`` when ``TP + FP`` is zero.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or differ in length.
    """
    y_true_array, y_pred_array = _check_classification_targets(y_true, y_pred)
    true_positive, false_positive, _ = _binary_counts(
        y_true_array, y_pred_array, positive_label
    )

    denominator = true_positive + false_positive
    if denominator == 0:
        return float(zero_division)
    return float(true_positive / denominator)


def recall_score(y_true, y_pred, positive_label=1, zero_division=0.0) -> float:
    """Compute binary recall: ``TP / (TP + FN)``.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.
    positive_label
        The label treated as the positive class.
    zero_division
        Value returned when there are no actual positives (zero denominator).

    Returns
    -------
    float
        The recall, or ``zero_division`` when ``TP + FN`` is zero.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or differ in length.
    """
    y_true_array, y_pred_array = _check_classification_targets(y_true, y_pred)
    true_positive, _, false_negative = _binary_counts(
        y_true_array, y_pred_array, positive_label
    )

    denominator = true_positive + false_negative
    if denominator == 0:
        return float(zero_division)
    return float(true_positive / denominator)


def f1_score(y_true, y_pred, positive_label=1, zero_division=0.0) -> float:
    """Compute the binary F1 score: the harmonic mean of precision and recall.

    Parameters
    ----------
    y_true
        Array-like ground-truth labels.
    y_pred
        Array-like predicted labels.
    positive_label
        The label treated as the positive class.
    zero_division
        Value returned when both precision and recall are zero.

    Returns
    -------
    float
        The F1 score, or ``zero_division`` when ``precision + recall`` is zero.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or differ in length.
    """
    precision = precision_score(
        y_true, y_pred, positive_label=positive_label, zero_division=zero_division
    )
    recall = recall_score(
        y_true, y_pred, positive_label=positive_label, zero_division=zero_division
    )

    denominator = precision + recall
    if denominator == 0:
        return float(zero_division)
    return float(2.0 * precision * recall / denominator)
