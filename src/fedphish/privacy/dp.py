"""
Differential privacy mechanisms for federated learning.

Implements local DP with gradient clipping and noise addition.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GradientClipper:
    """Clip gradients to bound sensitivity."""

    def __init__(
        self,
        clipping_norm: float = 1.0,
        clipping_method: str = "flat",  # 'flat', 'layer', 'adaptive'
    ):
        """
        Initialize gradient clipper.

        Args:
            clipping_norm: Maximum L2 norm for gradients
            clipping_method: Clipping strategy
        """
        self.clipping_norm = clipping_norm
        self.clipping_method = clipping_method

    def clip_gradients(
        self,
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Clip gradients to bound sensitivity.

        Args:
            gradients: Gradient values

        Returns:
            Tuple of (clipped_gradients, clip_factor, original_norm)
        """
        if self.clipping_method == "flat":
            return self._flat_clip(gradients)
        elif self.clipping_method == "layer":
            return self._layer_clip(gradients)
        elif self.clipping_method == "adaptive":
            return self._adaptive_clip(gradients)
        else:
            raise ValueError(f"Unknown clipping method: {self.clipping_method}")

    def _flat_clip(
        self,
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Flat clipping - clip entire gradient tensor."""
        # Flatten and compute norm
        flat_grads = gradients.flatten()
        norm = np.linalg.norm(flat_grads)

        if norm > self.clipping_norm:
            clip_factor = self.clipping_norm / norm
            clipped_grads = gradients * clip_factor
        else:
            clip_factor = 1.0
            clipped_grads = gradients.copy()

        return clipped_grads, clip_factor, norm

    def _layer_clip(
        self,
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Layer-wise clipping - clip each layer separately."""
        clipped_grads = np.zeros_like(gradients)
        clip_factors = []

        # Assume gradients is a list of layer-wise arrays
        if isinstance(gradients, list):
            for i, grad in enumerate(gradients):
                norm = np.linalg.norm(grad)
                if norm > self.clipping_norm:
                    factor = self.clipping_norm / norm
                    clipped_grads[i] = grad * factor
                    clip_factors.append(factor)
                else:
                    clipped_grads[i] = grad.copy()
                    clip_factors.append(1.0)
        else:
            # Fallback to flat clipping
            return self._flat_clip(gradients)

        avg_clip_factor = np.mean(clip_factors)
        original_norm = np.linalg.norm(np.concatenate([g.flatten() for g in gradients]))

        return clipped_grads, avg_clip_factor, original_norm

    def _adaptive_clip(
        self,
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Adaptive clipping - adjust based on historical norms.

        Uses a running average to adjust clipping threshold.
        """
        if not hasattr(self, "_norm_history"):
            self._norm_history = []

        flat_grads = gradients.flatten()
        norm = np.linalg.norm(flat_grads)

        # Update history
        self._norm_history.append(norm)
        if len(self._norm_history) > 100:
            self._norm_history.pop(0)

        # Adaptive threshold based on 75th percentile
        adaptive_norm = np.percentile(self._norm_history, 75) if self._norm_history else self.clipping_norm

        if norm > adaptive_norm:
            clip_factor = adaptive_norm / norm
            clipped_grads = gradients * clip_factor
        else:
            clip_factor = 1.0
            clipped_grads = gradients.copy()

        return clipped_grads, clip_factor, norm


class DifferentialPrivacy:
    """
    Differential privacy mechanism for gradient updates.

    Implements (ε, δ)-DP via Gaussian mechanism.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        clipping_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        secure_rng: bool = False,
    ):
        """
        Initialize DP mechanism.

        Args:
            epsilon: Privacy parameter
            delta: Delta parameter
            sensitivity: Gradient sensitivity (after clipping)
            clipping_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier (computed from ε if None)
            secure_rng: Use cryptographically secure RNG
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clipping_norm = clipping_norm
        self.secure_rng = secure_rng

        # Compute noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

        # Initialize clipper
        self.clipper = GradientClipper(
            clipping_norm=clipping_norm,
            clipping_method="flat",
        )

        logger.info(
            f"Initialized DP: ε={epsilon}, δ={delta}, "
            f"noise_multiplier={self.noise_multiplier:.4f}"
        )

    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from ε, δ.

        Uses Gaussian mechanism: σ = sensitivity * sqrt(2*ln(1.25/δ)) / ε
        """
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """
        Add DP noise to gradients.

        Args:
            gradients: Gradient values

        Returns:
            Noisy gradients
        """
        # Clip gradients
        clipped_grads, clip_factor, norm = self.clipper.clip_gradients(gradients)

        # Add Gaussian noise
        scale = self.clipping_norm * self.noise_multiplier

        if self.secure_rng:
            # Use cryptographically secure random number generation
            # Generate secure random bytes and convert to Gaussian using Box-Muller transform
            import secrets
            flat_size = gradients.size
            # Generate pairs of uniform random bytes for Box-Muller
            n_pairs = (flat_size + 1) // 2
            random_bytes = secrets.token_bytes(n_pairs * 8)  # 4 bytes per float (32-bit)
            # Convert bytes to uniform [0,1)
            uniform1 = np.frombuffer(random_bytes[:n_pairs*4], dtype=np.uint32) / 4294967296.0
            uniform2 = np.frombuffer(random_bytes[n_pairs*4:], dtype=np.uint32) / 4294967296.0
            # Box-Muller transform to Gaussian
            gaussian1 = np.sqrt(-2 * np.log(uniform1 + 1e-10)) * np.cos(2 * np.pi * uniform2)
            gaussian2 = np.sqrt(-2 * np.log(uniform1 + 1e-10)) * np.sin(2 * np.pi * uniform2)
            # Interleave and take needed amount
            secure_gaussian = np.empty(flat_size)
            secure_gaussian[0::2] = gaussian1[:len(secure_gaussian[0::2])]
            secure_gaussian[1::2] = gaussian2[:len(secure_gaussian[1::2])]
            noise = (secure_gaussian.reshape(gradients.shape) * scale).astype(gradients.dtype)
        else:
            noise = np.random.normal(0, scale, gradients.shape)

        noisy_grads = clipped_grads + noise

        logger.debug(
            f"Added DP noise: norm={norm:.4f}, clip_factor={clip_factor:.4f}, "
            f"noise_scale={scale:.4f}"
        )

        return noisy_grads

    def compute_privacy_spent(
        self,
        num_steps: int,
        sampling_probability: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute total privacy spend for multiple steps.

        Uses moments accountant (simplified).

        Args:
            num_steps: Number of training steps
            sampling_probability: Probability of sampling each record

        Returns:
            Tuple of (epsilon_spent, delta)
        """
        # Simplified advanced composition
        # In practice, use TensorBoard Privacy Accountant or Opacus

        epsilon_spent = self.epsilon * np.sqrt(num_steps * sampling_probability)

        return epsilon_spent, self.delta


class PrivacyAccountant:
    """
    Track privacy spend across multiple training rounds.

    Uses RDP (Rényi Differential Privacy) accounting.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        alphas: Optional[list] = None,
    ):
        """
        Initialize privacy accountant.

        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta
            alphas: List of RDP orders to compute
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # Standard RDP orders
        self.alphas = alphas or [1 + x / 10.0 for x in range(1, 100)]

        self.history = []

    def compute_rdp(
        self,
        noise_multiplier: float,
        sampling_probability: float,
        num_steps: int,
    ) -> float:
        """
        Compute RDP (Rényi Differential Privacy).

        Args:
            noise_multiplier: Noise multiplier
            sampling_probability: Sampling probability
            num_steps: Number of steps

        Returns:
            RDP at optimal alpha
        """
        # Simplified RDP computation
        # In practice, use Google's DP library

        rdp_values = []
        for alpha in self.alphas:
            if alpha == 1:
                continue

            # RDP for Gaussian mechanism
            rdp = (
                sampling_probability**2 * num_steps * alpha /
                (2 * noise_multiplier**2)
            )
            rdp_values.append(rdp)

        # Return maximum RDP
        return max(rdp_values) if rdp_values else 0

    def rdp_to_dp(
        self,
        rdp: float,
        delta: float,
    ) -> float:
        """
        Convert RDP to (ε, δ)-DP.

        Args:
            rdp: RDP value
            delta: Target delta

        Returns:
            Epsilon value
        """
        # Find optimal alpha
        epsilons = []
        for alpha in self.alphas:
            if alpha == 1:
                continue
            epsilon = rdp + np.log(1 / delta) / (alpha - 1)
            epsilons.append(epsilon)

        return min(epsilons) if epsilons else 0

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get total privacy spent so far.

        Returns:
            Tuple of (epsilon, delta)
        """
        total_rdp = sum(
            step["rdp"] for step in self.history if "rdp" in step
        )

        epsilon = self.rdp_to_dp(total_rdp, self.target_delta)

        return epsilon, self.target_delta

    def record_step(
        self,
        noise_multiplier: float,
        sampling_probability: float,
        num_steps: int,
    ):
        """
        Record a training step.

        Args:
            noise_multiplier: Noise multiplier used
            sampling_probability: Sampling probability
            num_steps: Number of steps
        """
        rdp = self.compute_rdp(noise_multiplier, sampling_probability, num_steps)

        self.history.append({
            "noise_multiplier": noise_multiplier,
            "sampling_probability": sampling_probability,
            "num_steps": num_steps,
            "rdp": rdp,
        })

        epsilon, delta = self.get_privacy_spent()

        logger.debug(
            f"Recorded privacy step: ε={epsilon:.4f}, δ={delta:.2e}, "
            f"noise_mult={noise_multiplier:.4f}"
        )

    def get_ledger(self) -> list:
        """Get privacy ledger."""
        epsilon, delta = self.get_privacy_spent()

        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": epsilon,
            "current_delta": delta,
            "steps": len(self.history),
            "history": self.history,
        }


class AdaptivePrivacy:
    """
    Adaptive privacy that adjusts noise based on gradient magnitude.

    Adds less noise when gradients are small (fast convergence).
    """

    def __init__(
        self,
        base_epsilon: float = 1.0,
        delta: float = 1e-5,
        min_noise_multiplier: float = 0.5,
        max_noise_multiplier: float = 2.0,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize adaptive privacy.

        Args:
            base_epsilon: Base privacy parameter
            delta: Delta parameter
            min_noise_multiplier: Minimum noise multiplier
            max_noise_multiplier: Maximum noise multiplier
            adaptation_rate: Rate of adaptation
        """
        self.base_epsilon = base_epsilon
        self.delta = delta
        self.min_noise_multiplier = min_noise_multiplier
        self.max_noise_multiplier = max_noise_multiplier
        self.adaptation_rate = adaptation_rate

        self.current_noise_multiplier = 1.0

    def compute_adaptive_noise(
        self,
        gradients: np.ndarray,
        target_norm: float = 1.0,
    ) -> np.ndarray:
        """
        Add adaptive noise to gradients.

        Args:
            gradients: Gradient values
            target_norm: Target gradient norm

        Returns:
            Noisy gradients
        """
        # Compute gradient norm
        grad_norm = np.linalg.norm(gradients.flatten())

        # Adjust noise multiplier based on gradient norm
        if grad_norm < target_norm:
            # Smaller gradients -> less noise
            self.current_noise_multiplier = max(
                self.min_noise_multiplier,
                self.current_noise_multiplier * (1 - self.adaptation_rate),
            )
        else:
            # Larger gradients -> more noise
            self.current_noise_multiplier = min(
                self.max_noise_multiplier,
                self.current_noise_multiplier * (1 + self.adaptation_rate),
            )

        # Add noise
        scale = grad_norm * self.current_noise_multiplier
        noise = np.random.normal(0, scale, gradients.shape)
        noisy_grads = gradients + noise

        logger.debug(
            f"Adaptive noise: grad_norm={grad_norm:.4f}, "
            f"noise_mult={self.current_noise_multiplier:.4f}"
        )

        return noisy_grads

    def get_effective_epsilon(self, num_steps: int) -> float:
        """
        Get effective epsilon based on average noise multiplier.

        Args:
            num_steps: Number of steps

        Returns:
            Effective epsilon
        """
        avg_noise_mult = self.current_noise_multiplier  # Simplified
        return self.base_epsilon / avg_noise_mult
