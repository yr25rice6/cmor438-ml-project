import numpy as np
import pytest

from cmor438_ml.models import KNNClassifier
from cmor438_ml.validation import NotFittedError


# --- constructor validation -------------------------------------------------

def test_default_n_neighbors_is_five():
    assert KNNClassifier().n_neighbors == 5


def test_accepts_valid_positive_integer():
    assert KNNClassifier(n_neighbors=3).n_neighbors == 3


@pytest.mark.parametrize("bad_value", [0, -1, 2.0, "3", True, False])
def test_rejects_invalid_n_neighbors(bad_value):
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=bad_value)


# --- fit --------------------------------------------------------------------

def _simple_dataset():
    # Two clearly separable clusters: small values are class 0, large are class 1.
    X = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [10.0, 10.0],
            [10.5, 9.5],
            [9.5, 10.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_fit_returns_self():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=3)
    assert model.fit(X, y) is model


def test_fit_stores_training_attributes():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=3).fit(X, y)
    assert np.array_equal(model.X_train_, X)
    assert np.array_equal(model.y_train_, y)
    assert model.n_features_in_ == 2
    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_fit_rejects_n_neighbors_larger_than_sample_count():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=4).fit(X, y)


def test_fit_rejects_mismatched_lengths():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=1).fit(X, y)


# --- predict ----------------------------------------------------------------

def test_predict_before_fit_raises():
    model = KNNClassifier()
    with pytest.raises(NotFittedError):
        model.predict(np.array([[0.0, 0.0]]))


def test_predict_on_separable_dataset():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=3).fit(X, y)
    X_test = np.array([[0.25, 0.25], [10.0, 10.0]])
    predictions = model.predict(X_test)
    assert np.array_equal(predictions, np.array([0, 1]))


def test_predict_with_string_labels():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["near", "near", "far", "far"])
    model = KNNClassifier(n_neighbors=1).fit(X, y)
    predictions = model.predict(np.array([[0.5], [10.5]]))
    assert np.array_equal(predictions, np.array(["near", "far"]))


def test_predict_breaks_ties_by_sorted_label_order():
    # Two equidistant neighbors with different labels; the tie must resolve to
    # the smaller label under np.unique's sorted order (0 before 1).
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 0])
    model = KNNClassifier(n_neighbors=2).fit(X, y)
    prediction = model.predict(np.array([[1.0, 0.0]]))
    assert np.array_equal(prediction, np.array([0]))


def test_predict_rejects_wrong_column_count():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=3).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_predict_returns_one_dimensional_array():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=3).fit(X, y)
    X_test = np.array([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5]])
    predictions = model.predict(X_test)
    assert predictions.ndim == 1
    assert predictions.shape == (3,)


# --- score ------------------------------------------------------------------

def test_score_on_deterministic_example():
    X, y = _simple_dataset()
    model = KNNClassifier(n_neighbors=1).fit(X, y)
    # Three queries: two land in their true cluster, one is mislabeled on purpose.
    X_test = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
    y_test = np.array([0, 1, 0])
    # Predictions are [0, 1, 1]; 2 of 3 match -> accuracy 2/3.
    assert model.score(X_test, y_test) == pytest.approx(2.0 / 3.0)


# --- robustness -------------------------------------------------------------

def test_fit_stores_copies_so_mutation_does_not_change_predictions():
    X, y = _simple_dataset()
    X_mutable = X.copy()
    y_mutable = y.copy()
    model = KNNClassifier(n_neighbors=3).fit(X_mutable, y_mutable)

    X_test = np.array([[0.25, 0.25], [10.0, 10.0]])
    before = model.predict(X_test)

    # Corrupt the original arrays after fitting.
    X_mutable[:] = 0.0
    y_mutable[:] = 0

    after = model.predict(X_test)
    assert np.array_equal(before, after)
    assert np.array_equal(after, np.array([0, 1]))
