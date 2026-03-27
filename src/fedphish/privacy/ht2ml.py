"""
HT2ML: Hybrid TEE and HE for Privacy-Preserving ML.

Combines homomorphic encryption for linear operations
with TEE for non-linear operations.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .he import SecureAggregator, TenSEALContext, he_available
from .tee import TrustedAggregator, create_tee_aggregator

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy levels for HT2ML."""

    LEVEL_1 = 1  # Local DP only
    LEVEL_2 = 2  # Local DP + Secure Aggregation (HE)
    LEVEL_3 = 3  # Full HT2ML (DP + HE + TEE)


class HT2MLAggregator:
    """
    Hybrid HE + TEE aggregator.

    Strategy:
    - Linear operations (averaging, weighted sums): HE
    - Non-linear operations (geometric median, Krum): TEE
    """

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.LEVEL_3,
        he_context: Optional[TenSEALContext] = None,
        use_real_tee: bool = False,
    ):
        """
        Initialize HT2ML aggregator.

        Args:
            privacy_level: Privacy level to use
            he_context: TenSEAL context (created if None)
            use_real_tee: Whether to use real TEE
        """
        self.privacy_level = privacy_level
        self.he_context = he_context or TenSEALContext()

        # Initialize components based on privacy level
        if privacy_level in [PrivacyLevel.LEVEL_2, PrivacyLevel.LEVEL_3]:
            if not he_available():
                logger.warning("TenSEAL not available, falling back to level 1")
                self.privacy_level = PrivacyLevel.LEVEL_1
                self.he_aggregator = None
            else:
                self.he_aggregator = SecureAggregator(self.he_context.context)
        else:
            self.he_aggregator = None

        if privacy_level == PrivacyLevel.LEVEL_3:
            self.tee_aggregator = create_tee_aggregator(use_real_tee)
        else:
            self.tee_aggregator = None

        logger.info(
            f"Initialized HT2ML aggregator at privacy level {privacy_level.name}"
        )

    def aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
        operation: str = "average",
        encrypted_updates: Optional[List] = None,
    ) -> np.ndarray:
        """
        Aggregate updates using appropriate privacy mechanism.

        Args:
            updates: List of gradient updates (plain)
            weights: Optional aggregation weights
            operation: Aggregation operation
            encrypted_updates: Pre-encrypted updates (for level 2+)

        Returns:
            Aggregated result
        """
        if self.privacy_level == PrivacyLevel.LEVEL_1:
            # Plain aggregation with local DP (assumed applied at client)
            return self._plain_aggregate(updates, weights, operation)

        elif self.privacy_level == PrivacyLevel.LEVEL_2:
            # HE-based aggregation
            if self._is_linear_operation(operation):
                return self._he_linear_aggregate(encrypted_updates, weights)
            else:
                # Fallback to plain for non-linear (no TEE at level 2)
                return self._plain_aggregate(updates, weights, operation)

        elif self.privacy_level == PrivacyLevel.LEVEL_3:
            # Full HT2ML: HE for linear, TEE for non-linear
            if self._is_linear_operation(operation):
                return self._he_linear_aggregate(encrypted_updates, weights)
            else:
                return self._tee_nonlinear_aggregate(updates, weights, operation)

        else:
            raise ValueError(f"Unknown privacy level: {self.privacy_level}")

    def _is_linear_operation(self, operation: str) -> bool:
        """Check if operation is linear (can use HE)."""
        linear_ops = ["average", "sum", "weighted_sum", "weighted_average"]
        return operation in linear_ops

    def _plain_aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]],
        operation: str,
    ) -> np.ndarray:
        """Plain aggregation (baseline, no HE/TEE)."""
        if operation == "average":
            return np.mean(updates, axis=0)
        elif operation == "sum":
            return np.sum(updates, axis=0)
        elif operation == "weighted_sum" or operation == "weighted_average":
            if weights is None:
                raise ValueError("Must provide weights for weighted aggregation")
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            return sum(w * u for w, u in zip(normalized_weights, updates))
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _he_linear_aggregate(
        self,
        encrypted_updates: List,
        weights: Optional[List[float]],
    ) -> np.ndarray:
        """Aggregate using HE (linear operations only)."""
        if self.he_aggregator is None:
            raise RuntimeError("HE aggregator not initialized")

        # Determine aggregation strategy
        if weights is None:
            strategy = "average"
        else:
            strategy = "weighted"

        # Aggregate encrypted
        encrypted_result = self.he_aggregator.aggregate(
            encrypted_updates,
            weights=weights,
        )

        # Decrypt
        result = self.he_aggregator.decrypt_aggregate(encrypted_result)

        return result

    def _tee_nonlinear_aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]],
        operation: str,
    ) -> np.ndarray:
        """Aggregate using TEE (non-linear operations)."""
        if self.tee_aggregator is None:
            raise RuntimeError("TEE aggregator not initialized")

        # Map operation name
        op_map = {
            "geomedian": "geomedian",
            "trimmed_mean": "trimmed_mean",
            "krum": "krum",
        }

        if operation not in op_map:
            raise ValueError(f"Unsupported non-linear operation: {operation}")

        return self.tee_aggregator.aggregate_nonlinear(
            updates,
            weights=weights,
            operation=op_map[operation],
        )


class PartitionStrategy:
    """
    Strategy for partitioning operations between HE and TEE.

    Analyzes computation graph to determine optimal partitioning.
    """

    @staticmethod
    def analyze_operations(
        operations: List[str],
    ) -> Dict[str, List[str]]:
        """
        Analyze operations and categorize as HE-compatible or TEE-only.

        Args:
            operations: List of operations

        Returns:
            Dictionary with 'he' and 'tee' operation lists
        """
        linear_ops = []
        nonlinear_ops = []

        for op in operations:
            # Determine if linear
            if op in ["average", "sum", "weighted_sum", "weighted_average", "add", "mul_scalar"]:
                linear_ops.append(op)
            else:
                nonlinear_ops.append(op)

        return {
            "he": linear_ops,
            "tee": nonlinear_ops,
        }

    @staticmethod
    def estimate_cost(
        operations: List[str],
        num_clients: int,
        gradient_size: int,
    ) -> Dict[str, float]:
        """
        Estimate computational cost for HE vs TEE.

        Args:
            operations: List of operations
            num_clients: Number of clients
            gradient_size: Size of gradient vectors

        Returns:
            Dictionary with cost estimates (ms)
        """
        # Approximate costs (these would be benchmarked in practice)
        he_cost_per_param = 0.01  # ms per parameter for HE
        tee_cost_per_op = 10.0  # ms overhead for TEE

        linear_count = sum(
            1 for op in operations
            if op in ["average", "sum", "weighted_sum", "weighted_average"]
        )
        nonlinear_count = len(operations) - linear_count

        he_cost = linear_count * he_cost_per_param * gradient_size * num_clients
        tee_cost = nonlinear_count * tee_cost_per_op

        return {
            "he_cost_ms": he_cost,
            "tee_cost_ms": tee_cost,
            "total_cost_ms": he_cost + tee_cost,
        }


class HT2MLClient:
    """
    Client-side HT2ML operations.

    Encrypts data before sending to server.
    """

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.LEVEL_3,
        he_context: Optional[TenSEALContext] = None,
    ):
        """
        Initialize HT2ML client.

        Args:
            privacy_level: Privacy level to use
            he_context: TenSEAL context
        """
        self.privacy_level = privacy_level
        self.he_context = he_context or TenSEALContext()

    def prepare_update(
        self,
        update: np.ndarray,
        apply_dp: bool = True,
        epsilon: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Prepare update for transmission.

        Args:
            update: Gradient update
            apply_dp: Whether to apply local DP
            epsilon: DP epsilon if applying DP

        Returns:
            Tuple of (plain_update, encrypted_update)
        """
        # Apply local DP if requested (always done at client)
        if apply_dp:
            from .dp import DifferentialPrivacy
            dp = DifferentialPrivacy(epsilon=epsilon)
            update = dp.add_noise(update)

        # Encrypt if using level 2 or 3
        encrypted_update = None
        if self.privacy_level in [PrivacyLevel.LEVEL_2, PrivacyLevel.LEVEL_3]:
            if he_available():
                from .he import EncryptedGradient
                encrypted_update = EncryptedGradient(
                    self.he_context.context,
                    update,
                )
            else:
                logger.warning("HE not available, sending plain update")

        return update, encrypted_update

    def get_public_key(self) -> bytes:
        """
        Get public key for encryption.

        Returns:
            Serialized public key
        """
        # For CKKS, the context contains necessary params
        return self.he_context.serialize()


class HT2MLMetrics:
    """Track HT2ML-specific metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.he_operations = []
        self.tee_operations = []
        self.operation_latencies = []
        self.privacy_level_usage = {level: 0 for level in PrivacyLevel}

    def log_he_operation(
        self,
        operation: str,
        time_ms: float,
    ):
        """Log HE operation."""
        self.he_operations.append(operation)
        self.operation_latencies.append(("he", operation, time_ms))

    def log_tee_operation(
        self,
        operation: str,
        time_ms: float,
    ):
        """Log TEE operation."""
        self.tee_operations.append(operation)
        self.operation_latencies.append(("tee", operation, time_ms))

    def log_privacy_level(self, level: PrivacyLevel):
        """Log privacy level usage."""
        self.privacy_level_usage[level] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        he_times = [t for t, op, time in self.operation_latencies if t == "he"]
        tee_times = [t for t, op, time in self.operation_latencies if t == "tee"]

        return {
            "num_he_operations": len(self.he_operations),
            "num_tee_operations": len(self.tee_operations),
            "avg_he_time_ms": np.mean(he_times) if he_times else 0,
            "avg_tee_time_ms": np.mean(tee_times) if tee_times else 0,
            "total_time_ms": sum(self.operation_latencies[i][2] for i in range(len(self.operation_latencies))) if self.operation_latencies else 0,
            "privacy_level_counts": {
                level.name: count
                for level, count in self.privacy_level_usage.items()
            },
        }


def create_ht2ml_aggregator(
    privacy_level: int = 3,
    **kwargs,
) -> HT2MLAggregator:
    """
    Create HT2ML aggregator.

    Args:
        privacy_level: Privacy level (1, 2, or 3)
        **kwargs: Additional arguments

    Returns:
        HT2MLAggregator instance
    """
    level = PrivacyLevel(privacy_level)
    return HT2MLAggregator(privacy_level=level, **kwargs)
