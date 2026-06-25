"""Tests for the Breast Cancer Wisconsin dataset loader.

These tests verify the shape, dtype, and structure of the data returned by
``load_breast_cancer_data`` in both its array and DataFrame forms, and that the
``return_frame`` flag is validated.
"""

import numpy as np
import pandas as pd
import pytest

from cmor438_ml.datasets import load_breast_cancer_data


def test_array_return_has_four_objects():
    result = load_breast_cancer_data(return_frame=False)
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_array_shapes_are_2d_and_1d():
    X, y, _, _ = load_breast_cancer_data(return_frame=False)
    assert X.ndim == 2
    assert y.ndim == 1


def test_X_and_y_have_same_number_of_samples():
    X, y, _, _ = load_breast_cancer_data(return_frame=False)
    assert X.shape[0] == y.shape[0]


def test_feature_names_match_column_count():
    X, _, feature_names, _ = load_breast_cancer_data(return_frame=False)
    assert len(feature_names) == X.shape[1]


def test_target_names_has_two_classes():
    _, _, _, target_names = load_breast_cancer_data(return_frame=False)
    assert len(target_names) == 2


def test_X_is_numeric():
    X, _, _, _ = load_breast_cancer_data(return_frame=False)
    assert np.issubdtype(X.dtype, np.number)


def test_y_has_two_unique_labels():
    _, y, _, _ = load_breast_cancer_data(return_frame=False)
    assert len(np.unique(y)) == 2


def test_return_frame_returns_dataframe():
    frame = load_breast_cancer_data(return_frame=True)
    assert isinstance(frame, pd.DataFrame)


def test_dataframe_has_target_column():
    frame = load_breast_cancer_data(return_frame=True)
    assert "target" in frame.columns


def test_dataframe_row_count_matches_array():
    _, y, _, _ = load_breast_cancer_data(return_frame=False)
    frame = load_breast_cancer_data(return_frame=True)
    assert frame.shape[0] == y.shape[0]


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_return_frame_rejects_non_bool(bad_value):
    with pytest.raises(ValueError):
        load_breast_cancer_data(return_frame=bad_value)
