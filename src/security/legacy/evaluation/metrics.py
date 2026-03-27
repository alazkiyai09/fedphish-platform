"""Evaluation metrics for co-evolution."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_attack_success_rate(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    attack_type: str,
    target_class: int = 1,
) -> float:
    """
    Compute attack success rate (ASR).

    Args:
        predictions: Model predictions
        true_labels: True labels
        attack_type: Type of attack
        target_class: Target class for backdoor

    Returns:
        Attack success rate
    """
    if attack_type == "backdoor":
        # ASR: proportion of triggered samples misclassified as target
        triggered_mask = true_labels == -1  # Special marker for triggered samples
        if not np.any(triggered_mask):
            return 0.0
        return np.mean(predictions[triggered_mask] == target_class)

    elif attack_type == "label_flip":
        # ASR: proportion of flipped labels resulting in misclassification
        return np.mean(predictions != true_labels)

    elif attack_type == "model_poisoning":
        # ASR: overall accuracy degradation
        accuracy = np.mean(predictions == true_labels)
        return 1.0 - accuracy

    elif attack_type == "evasion_poisoning":
        # ASR: evasion success rate
        return np.mean(predictions != true_labels)

    else:
        # Default: misclassification rate
        return np.mean(predictions != true_labels)


def compute_defense_overhead(
    defense_time: float,
    baseline_time: float,
) -> float:
    """
    Compute defense computational overhead.

    Args:
        defense_time: Time with defense
        baseline_time: Time without defense

    Returns:
        Overhead ratio
    """
    if baseline_time == 0:
        return 0.0
    return (defense_time - baseline_time) / baseline_time


def compute_attack_cost(
    attack_metadata: Dict[str, float],
    cost_model: str = "computation_only",
) -> float:
    """
    Compute attacker's computational cost.

    Args:
        attack_metadata: Attack execution metadata
        cost_model: Cost model ("computation_only", "with_exposure_risk")

    Returns:
        Computed cost
    """
    # Base computational cost
    cost = attack_metadata.get("strength", 1.0)

    if cost_model == "with_exposure_risk":
        # Add exposure risk
        exposure_risk = attack_metadata.get("exposure_risk", 0.0)
        cost += exposure_risk * 10.0

    return cost


def compute_defender_cost(
    defense_metadata: Dict[str, float],
) -> float:
    """
    Compute defender's computational cost.

    Args:
        defense_metadata: Defense execution metadata

    Returns:
        Computed cost
    """
    cost = 1.0  # Base cost

    if defense_metadata.get("used_pca", False):
        cost += 2.0
    if defense_metadata.get("used_clustering", False):
        cost += 3.0
    if defense_metadata.get("used_honeypots", False):
        num_honeypots = defense_metadata.get("num_honeypots", 0)
        cost += num_honeypots * 1.5

    return cost


def compute_detection_metrics(
    detected_malicious: List[int],
    actual_malicious: List[int],
    total_clients: int,
) -> Dict[str, float]:
    """
    Compute detection metrics.

    Args:
        detected_malicious: Client IDs detected as malicious
        actual_malicious: Actual malicious client IDs
        total_clients: Total number of clients

    Returns:
        Dictionary with detection_rate, false_positive_rate, precision, recall
    """
    # True positives
    tp = len(set(detected_malicious) & set(actual_malicious))

    # False positives
    fp = len(set(detected_malicious) - set(actual_malicious))

    # False negatives
    fn = len(set(actual_malicious) - set(detected_malicious))

    # True negatives
    tn = total_clients - tp - fp - fn

    # Detection rate (recall)
    detection_rate = tp / len(actual_malicious) if len(actual_malicious) > 0 else 0.0

    # False positive rate
    num_benign = total_clients - len(actual_malicious)
    false_positive_rate = fp / num_benign if num_benign > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (same as detection rate)
    recall = detection_rate

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def compute_statistical_significance(
    values1: List[float],
    values2: List[float],
    test: str = "ttest",
) -> Tuple[float, bool]:
    """
    Compute statistical significance test.

    Args:
        values1: First set of values
        values2: Second set of values
        test: Test type ("ttest", "wilcoxon")

    Returns:
        (p_value, significant) where significant is True if p < 0.05
    """
    from scipy import stats

    if test == "ttest":
        statistic, p_value = stats.ttest_ind(values1, values2)
    elif test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(values1, values2)
    else:
        raise ValueError(f"Unknown test: {test}")

    significant = p_value < 0.05

    logger.info(f"Statistical test ({test}): p={p_value:.4f}, significant={significant}")

    return p_value, significant
