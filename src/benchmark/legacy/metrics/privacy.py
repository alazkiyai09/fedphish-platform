"""Privacy metrics: epsilon, privacy loss, information leakage."""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_epsilon(
    num_steps: int,
    noise_multiplier: float,
    batch_size: int,
    num_samples: int,
    delta: float = 1e-5,
) -> float:
    """
    Compute epsilon for DP-SGD using moments accountant.

    Simplified version using standard formula.

    Args:
        num_steps: Number of training steps
        noise_multiplier: Noise multiplier (sigma)
        batch_size: Batch size
        num_samples: Total number of samples
        delta: Delta parameter

    Returns:
        Computed epsilon
    """
    # Simplified epsilon computation
    # In practice, use autodp or tensorflow-privacy for accurate computation

    # Sampling probability
    q = batch_size / num_samples

    # Noise standard deviation
    sigma = noise_multiplier

    # Simplified formula (conservative estimate)
    # This is an approximation
    epsilon = num_steps * q * q / (2 * sigma * sigma)
    epsilon += num_steps * q * (np.exp(1) - 1) / sigma

    return float(epsilon)


def compute_privacy_loss(
    epsilon: float,
    delta: float,
    target_epsilon: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute privacy loss metrics.

    Args:
        epsilon: Achieved epsilon
        delta: Delta value
        target_epsilon: Target epsilon (if specified)

    Returns:
        Privacy loss metrics
    """
    metrics = {
        "epsilon_achieved": epsilon,
        "delta": delta,
    }

    if target_epsilon is not None:
        metrics["epsilon_diff"] = abs(epsilon - target_epsilon)
        metrics["within_target"] = epsilon <= target_epsilon

    return metrics


def compute_information_leakage(
    gradients_a: np.ndarray,
    gradients_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """
    Compute information leakage through gradient inversion.

    Simplified metric based on gradient similarity.

    Args:
        gradients_a: Gradients from sample A
        gradients_b: Gradients from sample B
        labels_a: Labels for sample A
        labels_b: Labels for sample B

    Returns:
        Information leakage score
    """
    # Compute gradient similarity
    similarity = np.dot(gradients_a.flatten(), gradients_b.flatten())
    similarity /= (np.linalg.norm(gradients_a) * np.linalg.norm(gradients_b))

    # High similarity could indicate information leakage
    # when labels are different
    if labels_a != labels_b:
        return float(abs(similarity))
    else:
        return 0.0
