import numpy as np
import pytest

from cmor438_ml.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# --- accuracy_score ---------------------------------------------------------

def test_accuracy_score_perfect_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_score_mixed_predictions():
    # 3 of 4 labels match -> 0.75
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    assert accuracy_score(y_true, y_pred) == 0.75


def test_accuracy_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy_score([1, 0, 1], [1, 0])


def test_accuracy_score_rejects_two_dimensional_input():
    with pytest.raises(ValueError):
        accuracy_score([[1], [0]], [[1], [0]])


# --- confusion_matrix -------------------------------------------------------

def test_confusion_matrix_binary_default_labels():
    # labels default to sorted unique -> [0, 1]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    result = confusion_matrix(y_true, y_pred)
    expected = np.array([[1, 1], [0, 2]])
    assert np.array_equal(result, expected)


def test_confusion_matrix_custom_label_order():
    # explicit order [1, 0] flips rows/columns relative to the default
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    result = confusion_matrix(y_true, y_pred, labels=[1, 0])
    expected = np.array([[2, 0], [1, 1]])
    assert np.array_equal(result, expected)


def test_confusion_matrix_string_labels():
    y_true = ["cat", "dog", "cat"]
    y_pred = ["cat", "cat", "cat"]
    result = confusion_matrix(y_true, y_pred)  # sorted labels -> ["cat", "dog"]
    expected = np.array([[2, 0], [1, 0]])
    assert np.array_equal(result, expected)


def test_confusion_matrix_rejects_labels_omitting_observed_class():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, 1], labels=[0])


# --- precision_score / recall_score / f1_score ------------------------------

# Shared hand-computed example with positive_label=1:
#   y_true = [1, 0, 1, 1, 0, 0]
#   y_pred = [1, 1, 1, 0, 0, 1]
# TP = 2 (indices 0, 2), FP = 2 (indices 1, 5), FN = 1 (index 3)
#   precision = 2 / (2 + 2) = 0.5
#   recall    = 2 / (2 + 1) = 2/3
#   f1        = 2 * 0.5 * (2/3) / (0.5 + 2/3) = 4/7
_Y_TRUE = [1, 0, 1, 1, 0, 0]
_Y_PRED = [1, 1, 1, 0, 0, 1]


def test_precision_score_hand_computed():
    assert precision_score(_Y_TRUE, _Y_PRED) == pytest.approx(0.5)


def test_recall_score_hand_computed():
    assert recall_score(_Y_TRUE, _Y_PRED) == pytest.approx(2.0 / 3.0)


def test_f1_score_hand_computed():
    assert f1_score(_Y_TRUE, _Y_PRED) == pytest.approx(4.0 / 7.0)


def test_precision_score_zero_predicted_positives_uses_zero_division():
    # No predicted positives -> TP + FP == 0
    y_true = [1, 0]
    y_pred = [0, 0]
    assert precision_score(y_true, y_pred) == 0.0
    assert precision_score(y_true, y_pred, zero_division=-1.0) == -1.0


def test_recall_score_zero_actual_positives_uses_zero_division():
    # No actual positives -> TP + FN == 0
    y_true = [0, 0]
    y_pred = [0, 1]
    assert recall_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred, zero_division=-1.0) == -1.0


def test_f1_score_zero_precision_and_recall_uses_zero_division():
    # TP = 0 with a false positive and a false negative present, so precision
    # and recall are both genuine zeros: 0/(0+1). Then precision + recall == 0
    # and f1 falls back to zero_division.
    y_true = [1, 0]
    y_pred = [0, 1]
    assert f1_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred, zero_division=-1.0) == -1.0


# --- empty inputs -----------------------------------------------------------

def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError):
        accuracy_score([], [])
    with pytest.raises(ValueError):
        confusion_matrix([], [])
    with pytest.raises(ValueError):
        precision_score([], [])
    with pytest.raises(ValueError):
        recall_score([], [])
    with pytest.raises(ValueError):
        f1_score([], [])
