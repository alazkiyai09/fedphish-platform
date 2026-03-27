"""Fairness metrics: per-bank accuracy, variance."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def compute_per_bank_accuracy(
    model,
    client_data: List[tuple],
    client_ids: List[int]
) -> Dict[int, float]:
    """
    Compute per-bank (client) accuracy.

    Args:
        model: Trained model
        client_data: List of (X, y) tuples for each client
        client_ids: List of client IDs

    Returns:
        Dictionary mapping client_id to accuracy
    """
    per_bank_accuracy = {}

    for client_id, (X, y) in zip(client_ids, client_data):
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        per_bank_accuracy[client_id] = float(acc)

    return per_bank_accuracy


def compute_accuracy_variance(per_bank_accuracy: Dict[int, float]) -> Dict[str, float]:
    """
    Compute variance of per-bank accuracies.

    Args:
        per_bank_accuracy: Dictionary of per-bank accuracies

    Returns:
        Variance metrics
    """
    accuracies = list(per_bank_accuracy.values())

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "variance_accuracy": float(np.var(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "max_accuracy": float(np.max(accuracies)),
        "range_accuracy": float(np.max(accuracies) - np.min(accuracies)),
    }


def compute_fairness_metrics(
    model,
    client_data: List[tuple],
    client_ids: List[int],
    y_test_all: np.ndarray = None,
    predictions_all: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive fairness metrics.

    Args:
        model: Trained model
        client_data: List of (X, y) tuples for each client
        client_ids: List of client IDs
        y_test_all: All test labels (optional)
        predictions_all: All predictions (optional)

    Returns:
        Fairness metrics
    """
    # Per-bank accuracy
    per_bank_acc = compute_per_bank_accuracy(model, client_data, client_ids)

    # Variance metrics
    variance_metrics = compute_accuracy_variance(per_bank_acc)

    # Overall metrics
    metrics = variance_metrics.copy()

    # Add individual bank accuracies
    for bank_id, acc in per_bank_acc.items():
        metrics[f"bank_{bank_id}_accuracy"] = acc

    # Fairness ratio: min / max accuracy
    metrics["fairness_ratio"] = (
        metrics["min_accuracy"] / metrics["max_accuracy"]
        if metrics["max_accuracy"] > 0 else 0.0
    )

    # Coefficient of variation
    metrics["coefficient_of_variation"] = (
        metrics["std_accuracy"] / metrics["mean_accuracy"]
        if metrics["mean_accuracy"] > 0 else 0.0
    )

    return metrics


def compute_demographic_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray
) -> Dict[str, float]:
    """
    Compute demographic parity difference.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Sensitive attribute values

    Returns:
        Demographic parity metrics
    """
    unique_groups = np.unique(sensitive_attribute)

    # Positive prediction rate per group
    positive_rates = {}
    for group in unique_groups:
        group_mask = sensitive_attribute == group
        group_pred = y_pred[group_mask]
        positive_rate = np.mean(group_pred)
        positive_rates[group] = float(positive_rate)

    # Difference between max and min
    positive_rate_values = list(positive_rates.values())
    dp_difference = float(max(positive_rate_values) - min(positive_rate_values))

    return {
        "demographic_parity_difference": dp_difference,
        "positive_rates": positive_rates,
    }
