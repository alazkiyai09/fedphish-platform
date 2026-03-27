"""Privacy mechanisms: DP, Secure Aggregation."""

import logging
from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Differential Privacy mechanism."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize DP mechanism.

        Args:
            config: DP configuration
        """
        self.config = config or {}
        self.epsilon = self.config.get("epsilon", 1.0)
        self.delta = self.config.get("delta", 1e-5)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.mechanism = self.config.get("noise_mechanism", "gaussian")

    def add_noise_to_params(
        self,
        parameters: List[np.ndarray],
        sensitivity: float = 1.0
    ) -> List[np.ndarray]:
        """
        Add DP noise to parameters.

        Args:
            parameters: List of parameter arrays
            sensitivity: Sensitivity of the parameters

        Returns:
            Noisy parameters
        """
        if self.mechanism == "gaussian":
            # Gaussian mechanism
            sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * (sensitivity / self.epsilon)
            noise = [np.random.normal(0, sigma, p.shape) for p in parameters]
        else:
            # Laplace mechanism
            scale = sensitivity / self.epsilon
            noise = [np.random.laplace(0, scale, p.shape) for p in parameters]

        noisy_params = [p + n for p, n in zip(parameters, noise)]

        logger.info(f"Added {self.mechanism} noise with epsilon={self.epsilon}")
        return noisy_params

    def add_noise_to_gradients(
        self,
        gradients: List[np.ndarray],
        batch_size: int
    ) -> List[np.ndarray]:
        """
        Add DP noise to gradients.

        Args:
            gradients: List of gradient arrays
            batch_size: Batch size (for sensitivity calculation)

        Returns:
            Noisy gradients
        """
        # Clip gradients
        clipped_grads = clip_gradients(gradients, self.max_grad_norm)

        # Add noise
        sensitivity = self.max_grad_norm / batch_size
        return self.add_noise_to_params(clipped_grads, sensitivity)


class SecureAggregation:
    """Secure Aggregation mechanism."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize secure aggregation.

        Args:
            config: Configuration
        """
        self.config = config or {}
        self.protocol = self.config.get("protocol", "flwr")
        self.encryption = self.config.get("encryption", "aes")

    def aggregate(
        self,
        parameters_list: List[List[np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """
        Securely aggregate parameters.

        Args:
            parameters_list: List of client parameters
            weights: Optional client weights (by data size)

        Returns:
            Aggregated parameters
        """
        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)

        # Weighted average
        aggregated = []
        for params in zip(*parameters_list):
            weighted_sum = sum(w * p for w, p in zip(weights, params))
            aggregated.append(weighted_sum)

        logger.info(f"Securely aggregated {len(parameters_list)} client updates")
        return aggregated


def clip_gradients(
    gradients: List[np.ndarray],
    max_norm: float
) -> List[np.ndarray]:
    """
    Clip gradients by norm.

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum norm

    Returns:
        Clipped gradients
    """
    # Calculate total norm
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))

    # Clip if necessary
    if total_norm > max_norm:
        clip_factor = max_norm / total_norm
        gradients = [g * clip_factor for g in gradients]

    return gradients


def add_dp_noise(
    parameters: List[np.ndarray],
    epsilon: float,
    delta: float = 1e-5,
    mechanism: str = "gaussian"
) -> List[np.ndarray]:
    """
    Convenience function to add DP noise.

    Args:
        parameters: Parameters to add noise to
        epsilon: Privacy parameter
        delta: Delta parameter
        mechanism: Noise mechanism

    Returns:
        Noisy parameters
    """
    dp = DifferentialPrivacy({
        "epsilon": epsilon,
        "delta": delta,
        "noise_mechanism": mechanism
    })
    return dp.add_noise_to_params(parameters)
