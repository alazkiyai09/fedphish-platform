"""Classification metrics: accuracy, AUPRC, Recall@FPR."""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return float(accuracy_score(y_true, y_pred))


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        y_true: True labels
        y_scores: Prediction scores (probabilities)

    Returns:
        AUPRC score
    """
    return float(average_precision_score(y_true, y_scores))


def compute_recall_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.01
) -> float:
    """
    Compute recall at a specific false positive rate.

    Important for fraud detection where low FPR is critical.

    Args:
        y_true: True labels
        y_scores: Prediction scores
        target_fpr: Target FPR (default 1%)

    Returns:
        Recall at target FPR
    """
    # Get FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the index where FPR is closest to target
    idx = np.argmin(np.abs(fpr - target_fpr))

    # Return corresponding TPR (recall)
    return float(tpr[idx])


def compute_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores

    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "auprc": compute_auprc(y_true, y_scores),
        "recall_at_1_percent_fpr": compute_recall_at_fpr(y_true, y_scores, 0.01),
        "recall_at_5_percent_fpr": compute_recall_at_fpr(y_true, y_scores, 0.05),
        "recall_at_10_percent_fpr": compute_recall_at_fpr(y_true, y_scores, 0.10),
    }
