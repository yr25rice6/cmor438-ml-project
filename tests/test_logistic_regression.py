import numpy as np
import pytest

from cmor438_ml.models import LogisticRegressionGD
from cmor438_ml.validation import NotFittedError


# --- constructor validation -------------------------------------------------

def test_default_parameters_are_stored():
    model = LogisticRegressionGD()
    assert model.learning_rate == 0.1
    assert model.n_iterations == 1000
    assert model.fit_intercept is True
    assert model.tolerance is None


def test_accepts_valid_custom_parameters():
    model = LogisticRegressionGD(
        learning_rate=0.5, n_iterations=200, fit_intercept=False, tolerance=1e-4
    )
    assert model.learning_rate == 0.5
    assert model.n_iterations == 200
    assert model.fit_intercept is False
    assert model.tolerance == 1e-4


@pytest.mark.parametrize("bad_value", [0, -0.1, "0.1", True, False])
def test_rejects_invalid_learning_rate(bad_value):
    with pytest.raises(ValueError):
        LogisticRegressionGD(learning_rate=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, 2.0, "10", True, False])
def test_rejects_invalid_n_iterations(bad_value):
    with pytest.raises(ValueError):
        LogisticRegressionGD(n_iterations=bad_value)


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_rejects_invalid_fit_intercept(bad_value):
    # None is excluded as a valid value: fit_intercept must be a real bool.
    with pytest.raises(ValueError):
        LogisticRegressionGD(fit_intercept=bad_value)


@pytest.mark.parametrize("bad_value", [-1e-4, "0.0", True, False])
def test_rejects_invalid_tolerance(bad_value):
    with pytest.raises(ValueError):
        LogisticRegressionGD(tolerance=bad_value)


# --- shared datasets --------------------------------------------------------

def _separable_dataset():
    # One feature; small values are class 0, large values are class 1.
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def _separable_dataset_2d():
    # Two clearly separable clusters in 2-D.
    X = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 0.5],
            [8.0, 9.0],
            [9.0, 8.5],
            [9.5, 9.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# --- fit --------------------------------------------------------------------

def test_fit_returns_self():
    X, y = _separable_dataset()
    model = LogisticRegressionGD()
    assert model.fit(X, y) is model


def test_fit_stores_attributes():
    X, y = _separable_dataset_2d()
    model = LogisticRegressionGD(n_iterations=50).fit(X, y)
    assert model.coef_ is not None
    assert isinstance(model.intercept_, float)
    assert model.n_features_in_ == 2
    assert np.array_equal(model.classes_, np.array([0, 1]))
    assert model.n_iter_ == 50
    assert isinstance(model.loss_history_, list)
    assert len(model.loss_history_) == 50


def test_coef_has_feature_shape():
    X, y = _separable_dataset_2d()
    model = LogisticRegressionGD(n_iterations=10).fit(X, y)
    assert model.coef_.shape == (2,)


def test_classes_are_sorted():
    X = np.array([[10.0], [9.0], [1.0], [0.0]])
    y = np.array([1, 1, 0, 0])
    model = LogisticRegressionGD(n_iterations=10).fit(X, y)
    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_fit_rejects_one_class():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, y)


def test_fit_rejects_three_classes():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, y)


def test_fit_rejects_mismatched_lengths():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, y)


# --- predict_proba ----------------------------------------------------------

def test_predict_proba_before_fit_raises():
    model = LogisticRegressionGD()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.array([[0.0]]))


def test_predict_proba_returns_one_dimensional_array():
    X, y = _separable_dataset()
    model = LogisticRegressionGD(n_iterations=100).fit(X, y)
    proba = model.predict_proba(np.array([[0.0], [10.0], [5.0]]))
    assert proba.ndim == 1
    assert proba.shape == (3,)


def test_predict_proba_values_in_unit_interval():
    X, y = _separable_dataset()
    model = LogisticRegressionGD(n_iterations=100).fit(X, y)
    proba = model.predict_proba(np.array([[-50.0], [0.0], [5.0], [50.0]]))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_predict_proba_rejects_wrong_feature_count():
    X, y = _separable_dataset_2d()
    model = LogisticRegressionGD(n_iterations=10).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(np.array([[0.0, 0.0, 0.0]]))


# --- predict ----------------------------------------------------------------

def test_predict_on_separable_dataset():
    X, y = _separable_dataset()
    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=2000).fit(X, y)
    predictions = model.predict(np.array([[0.0], [10.0]]))
    assert np.array_equal(predictions, np.array([0, 1]))


def test_predict_with_string_labels():
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array(["low", "low", "low", "high", "high", "high"])
    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=2000).fit(X, y)
    # Sorted classes_ -> ["high", "low"]; class 1 is "low" (the larger string).
    predictions = model.predict(np.array([[0.0], [10.0]]))
    assert predictions.shape == (2,)
    assert predictions[0] == "low"
    assert predictions[1] == "high"


def test_predict_returns_one_dimensional_array():
    X, y = _separable_dataset_2d()
    model = LogisticRegressionGD(n_iterations=100).fit(X, y)
    X_test = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 1.0]])
    predictions = model.predict(X_test)
    assert predictions.ndim == 1
    assert predictions.shape == (3,)


# --- score ------------------------------------------------------------------

def test_score_high_accuracy_on_separable_dataset():
    X, y = _separable_dataset_2d()
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=2000).fit(X, y)
    assert model.score(X, y) >= 0.9


# --- training behavior ------------------------------------------------------

def test_loss_history_length_equals_n_iter():
    X, y = _separable_dataset()
    model = LogisticRegressionGD(n_iterations=123).fit(X, y)
    assert len(model.loss_history_) == model.n_iter_


def test_loss_decreases_over_training():
    X, y = _separable_dataset()
    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=500).fit(X, y)
    assert model.loss_history_[-1] <= model.loss_history_[0]


def test_tolerance_can_stop_early():
    X, y = _separable_dataset()
    # A large tolerance triggers early stopping before the iteration cap.
    model = LogisticRegressionGD(
        learning_rate=0.5, n_iterations=1000, tolerance=1.0
    ).fit(X, y)
    assert model.n_iter_ < 1000
    assert len(model.loss_history_) == model.n_iter_


def test_fitting_twice_gives_identical_predictions():
    X, y = _separable_dataset_2d()
    X_test = np.array([[0.0, 0.0], [9.0, 9.0], [4.0, 5.0]])
    first = LogisticRegressionGD(learning_rate=0.1, n_iterations=300).fit(X, y)
    second = LogisticRegressionGD(learning_rate=0.1, n_iterations=300).fit(X, y)
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
    assert np.allclose(first.coef_, second.coef_)
    assert first.intercept_ == second.intercept_


# --- robustness -------------------------------------------------------------

def test_fit_does_not_depend_on_later_mutation():
    X, y = _separable_dataset_2d()
    X_mutable = X.copy()
    y_mutable = y.copy()
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=300).fit(
        X_mutable, y_mutable
    )

    X_test = np.array([[0.0, 0.0], [9.0, 9.0]])
    before = model.predict(X_test)

    # Corrupt the original arrays after fitting.
    X_mutable[:] = 0.0
    y_mutable[:] = 0

    after = model.predict(X_test)
    assert np.array_equal(before, after)
