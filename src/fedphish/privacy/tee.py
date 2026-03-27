"""
Trusted Execution Environment (TEE) simulation.

Simulates TEE operations using Gramine or process isolation.
"""

import logging
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Gramine (optional)
try:
    import gramine  # type: ignore
    GRAMINE_AVAILABLE = True
except ImportError:
    GRAMINE_AVAILABLE = False
    gramine = None


class GramineTEE:
    """
    Simulate Gramine TEE environment.

    In production, this would run actual code inside SGX enclaves.
    For development/testing, we simulate with isolated processes.
    """

    def __init__(
        self,
        use_real_tee: bool = False,
        attestation_required: bool = True,
    ):
        """
        Initialize TEE simulator.

        Args:
            use_real_tee: Whether to use real SGX (requires Gramine + SGX hardware)
            attestation_required: Whether to perform remote attestation
        """
        self.use_real_tee = use_real_tee and GRAMINE_AVAILABLE
        self.attestation_required = attestation_required

        if self.use_real_tee:
            logger.info("Using real Gramine TEE")
        else:
            logger.info("Using TEE simulation (process isolation)")

    def execute_in_trusted_environment(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function in trusted environment.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if self.use_real_tee:
            return self._execute_in_gramine(func, *args, **kwargs)
        else:
            return self._execute_in_isolated_process(func, *args, **kwargs)

    def _execute_in_gramine(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute in real Gramine TEE."""
        # In production, this would use Gramine to run in SGX
        # For now, fallback to simulation
        logger.warning("Real Gramine TEE not fully implemented, using simulation")
        return self._execute_in_isolated_process(func, *args, **kwargs)

    def _execute_in_isolated_process(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute in isolated process (simulates TEE)."""
        # Use multiprocessing for isolation
        with mp.Pool(1) as pool:
            result = pool.apply(func, args, kwargs)

        return result


class TrustedAggregator:
    """
    Perform trusted aggregation inside TEE.

    Used for non-linear operations that cannot be done with HE.
    """

    def __init__(
        self,
        tee: Optional[GramineTEE] = None,
    ):
        """
        Initialize trusted aggregator.

        Args:
            tee: TEE instance (created if None)
        """
        self.tee = tee or GramineTEE(use_real_tee=False)
        self.aggregation_count = 0

    def aggregate_nonlinear(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
        operation: str = "geomedian",  # 'geomedian', 'trimmed_mean', 'krum'
    ) -> np.ndarray:
        """
        Perform non-linear aggregation inside TEE.

        Args:
            updates: List of gradient updates
            weights: Optional weights (not used for all operations)
            operation: Aggregation operation

        Returns:
            Aggregated result
        """
        self.aggregation_count += 1

        if operation == "geomedian":
            return self._geometric_median_trusted(updates)
        elif operation == "trimmed_mean":
            return self._trimmed_mean_trusted(updates)
        elif operation == "krum":
            return self._krum_trusted(updates)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _geometric_median_trusted(
        self,
        updates: List[np.ndarray],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute geometric median (Weiszfeld algorithm) inside TEE.

        More robust to outliers than mean.

        Args:
            updates: List of updates
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Geometric median
        """
        # Execute in trusted environment
        def compute_median(updates_list):
            # Initialize with mean
            median = np.mean(updates_list, axis=0)

            for i in range(max_iter):
                # Compute distances
                distances = np.array([
                    np.linalg.norm(update - median) + 1e-10
                    for update in updates_list
                ])

                # Weiszfeld update
                weights = 1.0 / distances
                weights = weights / weights.sum()
                new_median = np.average(updates_list, axis=0, weights=weights)

                # Check convergence
                if np.linalg.norm(new_median - median) < tol:
                    break

                median = new_median

            return median

        # Run in TEE
        return self.tee.execute_in_trusted_environment(
            compute_median,
            updates,
        )

    def _trimmed_mean_trusted(
        self,
        updates: List[np.ndarray],
        trim_fraction: float = 0.2,
    ) -> np.ndarray:
        """
        Compute trimmed mean inside TEE.

        Removes extreme values before averaging.

        Args:
            updates: List of updates
            trim_fraction: Fraction to trim from each end

        Returns:
            Trimmed mean
        """
        def compute_trimmed_mean(updates_list, trim_frac):
            # Compute norms for sorting
            norms = np.array([np.linalg.norm(u) for u in updates_list])

            # Sort by norms
            sorted_indices = np.argsort(norms)

            # Trim
            n = len(updates_list)
            k = int(n * trim_frac)
            trimmed_indices = sorted_indices[k:n-k]

            # Average remaining
            trimmed_updates = [updates_list[i] for i in trimmed_indices]
            return np.mean(trimmed_updates, axis=0)

        return self.tee.execute_in_trusted_environment(
            compute_trimmed_mean,
            updates,
            trim_fraction,
        )

    def _krum_trusted(
        self,
        updates: List[np.ndarray],
        num_byzantines: int = 1,
    ) -> np.ndarray:
        """
        Krum aggregation inside TEE.

        Selects update closest to all others (robust to Byzantine).

        Args:
            updates: List of updates
            num_byzantines: Number of Byzantine clients

        Returns:
            Krum-selected update
        """
        def compute_krum(updates_list, n_byzantine):
            n = len(updates_list)

            # Compute pairwise distances
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(updates_list[i] - updates_list[j])**2
                    distances[i, j] = dist
                    distances[j, i] = dist

            # Compute scores for each update
            scores = []
            for i in range(n):
                # Sort distances, take n - byzantine - 2 smallest
                sorted_dists = np.sort(distances[i])
                score = np.sum(sorted_dists[:n - n_byzantine - 2])
                scores.append(score)

            # Return update with minimum score
            best_idx = np.argmin(scores)
            return updates_list[best_idx]

        return self.tee.execute_in_trusted_environment(
            compute_krum,
            updates,
            num_byzantines,
        )


class Attestation:
    """
    Remote attestation for TEE.

    Verifies that code runs in genuine TEE.
    """

    def __init__(
        self,
        tee: Optional[GramineTEE] = None,
    ):
        """
        Initialize attestation manager.

        Args:
            tee: TEE instance
        """
        self.tee = tee or GramineTEE()
        self.verified = False

    def generate_attestation(self) -> Dict[str, Any]:
        """
        Generate attestation report.

        Returns:
            Attestation report with measurement
        """
        # In real SGX, this would generate quote with report
        # For simulation, generate a mock report

        import hashlib
        import time

        # Create measurement hash
        measurement_data = f"fedphish-tee-{time.time()}".encode()
        measurement = hashlib.sha256(measurement_data).hexdigest()

        report = {
            "measurement": measurement,
            "timestamp": time.time(),
            "tee_type": "SGX" if self.tee.use_real_tee else "SIMULATED",
            "status": "VERIFIED" if self.tee.use_real_tee else "SIMULATED",
        }

        logger.info(f"Generated attestation: {report['status']}")

        return report

    def verify_attestation(
        self,
        report: Dict[str, Any],
        expected_measurement: Optional[str] = None,
    ) -> bool:
        """
        Verify attestation report.

        Args:
            report: Attestation report to verify
            expected_measurement: Expected measurement hash

        Returns:
            True if verified
        """
        # In production, verify with Intel IAS
        # For simulation, basic checks

        if report.get("status") == "SIMULATED":
            logger.warning("Simulated attestation - skipping verification")
            self.verified = True
            return True

        if expected_measurement and report.get("measurement") != expected_measurement:
            logger.error("Measurement mismatch")
            return False

        self.verified = True
        logger.info("Attestation verified successfully")

        return True


class TEEMetrics:
    """Track TEE-related metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.execution_times = []
        self.attestation_times = []
        self.memory_usage = []
        self.aggregation_count = 0

    def log_execution(
        self,
        time_sec: float,
        memory_mb: float,
    ):
        """Log execution metrics."""
        self.execution_times.append(time_sec)
        self.memory_usage.append(memory_mb)

    def log_attestation(self, time_sec: float):
        """Log attestation time."""
        self.attestation_times.append(time_sec)

    def increment_aggregation(self):
        """Increment aggregation counter."""
        self.aggregation_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_aggregations": self.aggregation_count,
            "avg_execution_time_ms": np.mean(self.execution_times) * 1000 if self.execution_times else 0,
            "avg_attestation_time_ms": np.mean(self.attestation_times) * 1000 if self.attestation_times else 0,
            "avg_memory_mb": np.mean(self.memory_usage) if self.memory_usage else 0,
        }


def tee_available() -> bool:
    """Check if TEE is available."""
    return GRAMINE_AVAILABLE


def create_tee_aggregator(
    use_real_tee: bool = False,
) -> TrustedAggregator:
    """
    Create TEE aggregator.

    Args:
        use_real_tee: Whether to use real TEE

    Returns:
        TrustedAggregator instance
    """
    tee = GramineTEE(use_real_tee=use_real_tee)
    return TrustedAggregator(tee=tee)
