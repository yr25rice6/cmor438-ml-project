"""cmor438_ml: reusable machine learning utilities for the CMOR 438 / INDE 577 final project."""

from cmor438_ml.datasets import load_breast_cancer_data
from cmor438_ml.evaluation import (
    classification_report_dict,
    compare_classifiers,
    evaluate_classifier,
)
from cmor438_ml.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from cmor438_ml.model_selection import (
    cross_validate_classifier,
    k_fold_split,
)
from cmor438_ml.preprocessing import (
    StandardScalerScratch,
    add_intercept,
    train_test_split,
)
from cmor438_ml.validation import (
    NotFittedError,
    check_array_2d,
    check_is_fitted,
    check_X_y,
    ensure_binary_labels,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # validation
    "NotFittedError",
    "check_array_2d",
    "check_X_y",
    "check_is_fitted",
    "ensure_binary_labels",
    # metrics
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    # preprocessing
    "train_test_split",
    "StandardScalerScratch",
    "add_intercept",
    # datasets
    "load_breast_cancer_data",
    # evaluation
    "classification_report_dict",
    "evaluate_classifier",
    "compare_classifiers",
    # model selection
    "k_fold_split",
    "cross_validate_classifier",
]
