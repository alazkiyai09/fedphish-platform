"""
Homomorphic encryption for federated learning.

Implements secure aggregation using TenSEAL (CKKS scheme).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    ts = None
    logging.warning("TenSEAL not available. Install with: pip install tenseal")

logger = logging.getLogger(__name__)


class TenSEALContext:
    """
    Manage TenSEAL encryption context.

    Encapsulates CKKS encryption parameters.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        global_scale: float = 2**40,
        security_level: int = 128,
    ):
        """
        Initialize TenSEAL context.

        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Bit sizes for coefficient moduli
            global_scale: Scale for CKKS encoding
            security_level: Security level (128 or 192)
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is not installed")

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        self.security_level = security_level

        # Create context
        self.context = self._create_context()

        logger.info(
            f"Created TenSEAL context: poly_modulus={poly_modulus_degree}, "
            f"security_level={security_level}"
        )

    def _create_context(self) -> ts.Context:
        """Create and configure TenSEAL context."""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,
        )

        # Set scale
        context.global_scale = self.global_scale

        # Generate Galois keys (needed for ciphertext rotations)
        context.generate_galois_keys()

        return context

    def get_context(self) -> ts.Context:
        """Get the TenSEAL context."""
        return self.context

    def serialize(self) -> bytes:
        """Serialize context for transmission."""
        return self.context.serialize()

    @staticmethod
    def deserialize(data: bytes) -> ts.Context:
        """Deserialize context from bytes."""
        return ts.context_from(data)


class EncryptedGradient:
    """
    Encrypted gradient using CKKS.

    Wraps TenSEAL CKKS vector for gradient values.
    """

    def __init__(
        self,
        context: ts.Context,
        gradient: Optional[np.ndarray] = None,
        encrypted_vector: Optional[ts.CKKSVector] = None,
    ):
        """
        Initialize encrypted gradient.

        Args:
            context: TenSEAL context
            gradient: Plain gradient values (to encrypt)
            encrypted_vector: Pre-encrypted vector
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is not installed")

        self.context = context

        if encrypted_vector is not None:
            self.encrypted = encrypted_vector
        elif gradient is not None:
            self.encrypted = self._encrypt(gradient)
        else:
            raise ValueError("Must provide either gradient or encrypted_vector")

    def _encrypt(self, gradient: np.ndarray) -> ts.CKKSVector:
        """Encrypt gradient values."""
        # Flatten gradient
        flat_grad = gradient.flatten().astype(np.float64)

        # Encrypt
        encrypted = ts.ckks_vector(self.context, flat_grad.tolist())

        return encrypted

    def decrypt(self, secret_key: ts.SecretKey) -> np.ndarray:
        """
        Decrypt to get plain gradient values.

        Args:
            secret_key: Secret key for decryption

        Returns:
            Decrypted gradient as numpy array
        """
        decrypted = self.encrypted.decrypt(secret_key)
        return np.array(decrypted)

    def serialize(self) -> bytes:
        """Serialize encrypted gradient."""
        return self.encrypted.serialize()

    @staticmethod
    def deserialize(
        data: bytes,
        context: ts.Context,
    ) -> "EncryptedGradient":
        """Deserialize encrypted gradient."""
        encrypted = ts.ckks_vector_from(context, data)
        return EncryptedGradient(context=context, encrypted_vector=encrypted)

    def size(self) -> int:
        """Get size of encrypted gradient in bytes."""
        return len(self.serialize())

    def __add__(self, other: "EncryptedGradient") -> "EncryptedGradient":
        """Add two encrypted gradients (homomorphic operation)."""
        result = self.encrypted + other.encrypted
        return EncryptedGradient(
            context=self.context,
            encrypted_vector=result,
        )

    def __mul__(self, scalar: float) -> "EncryptedGradient":
        """Multiply encrypted gradient by scalar (plain)."""
        result = self.encrypted * scalar
        return EncryptedGradient(
            context=self.context,
            encrypted_vector=result,
        )


class SecureAggregator:
    """
    Securely aggregate encrypted gradients.

    Performs homomorphic aggregation without decrypting individual updates.
    """

    def __init__(
        self,
        context: ts.Context,
        aggregation_strategy: str = "average",  # 'average', 'sum', 'weighted'
    ):
        """
        Initialize secure aggregator.

        Args:
            context: TenSEAL context
            aggregation_strategy: How to combine gradients
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is not installed")

        self.context = context
        self.aggregation_strategy = aggregation_strategy

        # Store secret key if server (for final decryption)
        self.secret_key = context.secret_key()

        logger.info(f"Initialized secure aggregator: strategy={aggregation_strategy}")

    def aggregate(
        self,
        encrypted_gradients: List[EncryptedGradient],
        weights: Optional[List[float]] = None,
    ) -> EncryptedGradient:
        """
        Aggregate encrypted gradients.

        Args:
            encrypted_gradients: List of encrypted gradients from clients
            weights: Optional weights for weighted aggregation

        Returns:
            Aggregated encrypted gradient
        """
        if not encrypted_gradients:
            raise ValueError("No gradients to aggregate")

        if self.aggregation_strategy == "sum":
            return self._aggregate_sum(encrypted_gradients)
        elif self.aggregation_strategy == "average":
            return self._aggregate_average(encrypted_gradients)
        elif self.aggregation_strategy == "weighted":
            if weights is None:
                raise ValueError("Must provide weights for weighted aggregation")
            return self._aggregate_weighted(encrypted_gradients, weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

    def _aggregate_sum(
        self,
        encrypted_gradients: List[EncryptedGradient],
    ) -> EncryptedGradient:
        """Sum encrypted gradients."""
        # Start with first gradient
        result = encrypted_gradients[0]

        # Add remaining gradients
        for grad in encrypted_gradients[1:]:
            result = result + grad

        return result

    def _aggregate_average(
        self,
        encrypted_gradients: List[EncryptedGradient],
    ) -> EncryptedGradient:
        """Average encrypted gradients."""
        # Sum all gradients
        summed = self._aggregate_sum(encrypted_gradients)

        # Divide by count (multiply by 1/n)
        count = len(encrypted_gradients)
        averaged = summed * (1.0 / count)

        return averaged

    def _aggregate_weighted(
        self,
        encrypted_gradients: List[EncryptedGradient],
        weights: List[float],
    ) -> EncryptedGradient:
        """Weighted average of encrypted gradients."""
        if len(encrypted_gradients) != len(weights):
            raise ValueError("Number of gradients and weights must match")

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weight and sum
        result = encrypted_gradients[0] * normalized_weights[0]
        for grad, weight in zip(encrypted_gradients[1:], normalized_weights[1:]):
            result = result + (grad * weight)

        return result

    def decrypt_aggregate(
        self,
        encrypted_aggregate: EncryptedGradient,
    ) -> np.ndarray:
        """
        Decrypt aggregated gradient (server-side).

        Args:
            encrypted_aggregate: Aggregated encrypted gradient

        Returns:
            Decrypted aggregated gradient
        """
        return encrypted_aggregate.decrypt(self.secret_key)

    def verify_aggregate(
        self,
        encrypted_aggregate: EncryptedGradient,
    ) -> bool:
        """
        Verify encrypted aggregate is well-formed.

        Args:
            encrypted_aggregate: Aggregated encrypted gradient

        Returns:
            True if valid
        """
        try:
            # Try to access the encrypted vector
            _ = encrypted_aggregate.encrypted.size()
            return True
        except Exception as e:
            logger.error(f"Aggregate verification failed: {e}")
            return False


class HEPerformanceMetrics:
    """Track HE-related performance metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.encryption_times = []
        self.decryption_times = []
        self.aggregation_times = []
        self.communication_sizes = []
        self.computation_overheads = []

    def log_encryption(
        self,
        time_sec: float,
        size_bytes: int,
    ):
        """Log encryption metrics."""
        self.encryption_times.append(time_sec)
        self.communication_sizes.append(size_bytes)

    def log_decryption(
        self,
        time_sec: float,
    ):
        """Log decryption metrics."""
        self.decryption_times.append(time_sec)

    def log_aggregation(
        self,
        time_sec: float,
        overhead_ratio: float,  # HE time / plain time
    ):
        """Log aggregation metrics."""
        self.aggregation_times.append(time_sec)
        self.computation_overheads.append(overhead_ratio)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "avg_encryption_time_ms": np.mean(self.encryption_times) * 1000 if self.encryption_times else 0,
            "avg_decryption_time_ms": np.mean(self.decryption_times) * 1000 if self.decryption_times else 0,
            "avg_aggregation_time_ms": np.mean(self.aggregation_times) * 1000 if self.aggregation_times else 0,
            "avg_communication_kb": np.mean(self.communication_sizes) / 1024 if self.communication_sizes else 0,
            "avg_computation_overhead": np.mean(self.computation_overheads) if self.computation_overheads else 0,
        }


def he_available() -> bool:
    """Check if homomorphic encryption is available."""
    return TENSEAL_AVAILABLE


def simulate_he_aggregation(
    gradients: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Simulate HE aggregation (for testing when TenSEAL unavailable).

    Args:
        gradients: List of gradients
        weights: Optional weights

    Returns:
        Aggregated gradient
    """
    if not TENSEAL_AVAILABLE:
        logger.warning("TenSEAL not available, using plain aggregation")

    if weights is None:
        return np.mean(gradients, axis=0)
    else:
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        return sum(w * g for w, g in zip(normalized_weights, gradients))
