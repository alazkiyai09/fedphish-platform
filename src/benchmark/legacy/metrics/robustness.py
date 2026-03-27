"""Robustness metrics: attack success rate, accuracy degradation."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def compute_attack_success_rate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_type: str,
    **kwargs
) -> float:
    """
    Compute attack success rate.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        attack_type: Type of attack
        **kwargs: Additional attack parameters

    Returns:
        Attack success rate
    """
    if attack_type == "backdoor":
        # For backdoor, check if trigger causes target prediction
        trigger_pattern = kwargs.get("trigger_pattern", "https://secure-login")
        target_label = kwargs.get("target_label", 1)

        # Create test samples with trigger (simplified)
        # In practice, you'd insert trigger into actual test data
        y_pred = model.predict(X_test)

        # Approximate ASR: fraction of samples predicted as target label
        # when they should be something else
        asr = np.mean((y_pred == target_label) & (y_test != target_label))
        return float(asr)

    elif attack_type == "label_flip":
        # For label flip, ASR is fraction of flipped predictions
        clean_model = kwargs.get("clean_model")
        if clean_model is not None:
            y_pred_clean = clean_model.predict(X_test)
            y_pred_poisoned = model.predict(X_test)

            # ASR: fraction of predictions that changed
            asr = np.mean(y_pred_clean != y_pred_poisoned)
            return float(asr)
        else:
            return 0.0

    elif attack_type == "model_poisoning":
        # For model poisoning, ASR is based on accuracy degradation
        clean_accuracy = kwargs.get("clean_accuracy", 1.0)
        y_pred = model.predict(X_test)
        poisoned_accuracy = accuracy_score(y_test, y_pred)

        # ASR as relative degradation
        asr = (clean_accuracy - poisoned_accuracy) / clean_accuracy
        return float(asr)

    else:
        return 0.0


def compute_accuracy_degradation(
    clean_accuracy: float,
    poisoned_accuracy: float
) -> float:
    """
    Compute accuracy degradation.

    Args:
        clean_accuracy: Accuracy without attack
        poisoned_accuracy: Accuracy with attack

    Returns:
        Accuracy degradation (absolute and relative)
    """
    absolute = clean_accuracy - poisoned_accuracy
    relative = absolute / clean_accuracy if clean_accuracy > 0 else 0.0

    return {
        "absolute_degradation": float(absolute),
        "relative_degradation": float(relative),
        "clean_accuracy": float(clean_accuracy),
        "poisoned_accuracy": float(poisoned_accuracy),
    }


def compute_byzantine_resilience(
    client_updates: list,
    num_malicious: int,
    aggregated_result: np.ndarray
) -> Dict[str, float]:
    """
    Compute resilience to Byzantine attacks.

    Args:
        client_updates: List of client updates
        num_malicious: Number of malicious clients
        aggregated_result: Aggregated result

    Returns:
        Resilience metrics
    """
    # Compute average distance of malicious updates from aggregate
    distances = []
    for update in client_updates[-num_malicious:]:  # Last N are malicious
        distance = np.linalg.norm(update - aggregated_result)
        distances.append(distance)

    avg_distance = np.mean(distances) if distances else 0.0

    return {
        "avg_malicious_distance": float(avg_distance),
        "num_malicious": num_malicious,
    }
