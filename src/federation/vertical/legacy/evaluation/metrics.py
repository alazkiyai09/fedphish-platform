"""
Evaluation metrics for phishing detection models.

Implements accuracy, precision, recall, F1, AUC-ROC, etc.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    log_loss
)


def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except (ValueError, IndexError):
            # May fail if only one class present
            metrics['auc_roc'] = 0.0
            metrics['log_loss'] = float('inf')

    return metrics


def compute_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_per_class_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class (safe vs phishing) metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with metrics for each class
    """
    metrics = {}

    for class_label in [0, 1]:
        class_name = 'safe' if class_label == 0 else 'phishing'

        # Binary indicators for this class
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)

        metrics[class_name] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }

    return metrics
