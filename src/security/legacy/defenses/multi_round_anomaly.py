"""Multi-round anomaly detection defense."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .base import (
    BaseDefense,
    DefenseConfig,
    DefenseHistory,
    DefenderObservability,
)

logger = logging.getLogger(__name__)


class MultiRoundAnomalyDetection(BaseDefense):
    """
    Multi-round anomaly detection.

    Tracks client behavior over time to detect slow poisoning attacks.
    """

    def __init__(
        self,
        defender_observability: DefenderObservability,
        window_size: int = 10,
        threshold_method: str = "adaptive",
        baseline_method: str = "moving_average",
        static_threshold: float = 3.0,
        alpha: float = 0.1,
        percentile: int = 95,
    ):
        """
        Initialize multi-round anomaly detection.

        Args:
            defender_observability: Defender observability
            window_size: Window size for anomaly detection
            threshold_method: How to compute threshold ("adaptive", "static")
            baseline_method: How to compute baseline ("moving_average", "exponential", "percentile")
            static_threshold: Static threshold value
            alpha: Alpha for exponential moving average
            percentile: Percentile for percentile baseline
        """
        config = DefenseConfig(defense_type="multi_round_anomaly")
        super().__init__(defender_observability, config)

        self.window_size = window_size
        self.threshold_method = threshold_method
        self.baseline_method = baseline_method
        self.static_threshold = static_threshold
        self.alpha = alpha
        self.percentile = percentile

        # Adaptive threshold
        self.current_threshold = static_threshold

    def detect(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
        round_num: int,
        history: DefenseHistory,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Detect malicious clients using multi-round anomaly detection.

        Args:
            client_updates: List of (client_id, parameters) tuples
            round_num: Current round
            history: Defense history

        Returns:
            (malicious_client_ids, detection_metadata)
        """
        malicious_ids = []
        anomaly_scores = {}
        thresholds = {}

        for client_id, params in client_updates:
            # Compute anomaly score
            score = self.compute_anomaly_score(client_id, params, history)
            anomaly_scores[client_id] = score

            # Get threshold for this client
            threshold = self._get_threshold(client_id, history)
            thresholds[client_id] = threshold

            # Detect anomaly
            if score > threshold:
                malicious_ids.append(client_id)

            # Record behavior
            update_norm = self._compute_update_norm(params)
            history.add_client_behavior(
                client_id,
                round_num,
                {"anomaly_score": score, "update_norm": update_norm}
            )

        # Compute metadata
        metadata = {
            "anomaly_scores": anomaly_scores,
            "thresholds": thresholds,
            "num_malicious": len(malicious_ids),
            "threshold_method": self.threshold_method,
            "baseline_method": self.baseline_method,
        }

        logger.debug(
            f"Multi-round anomaly detection: detected {len(malicious_ids)} malicious clients"
        )

        return malicious_ids, metadata

    def compute_anomaly_score(
        self,
        client_id: int,
        update: Dict[str, torch.Tensor],
        history: DefenseHistory,
    ) -> float:
        """
        Compute anomaly score based on historical behavior.

        Args:
            client_id: Client ID
            update: Current update
            history: Defense history

        Returns:
            Anomaly score (higher = more anomalous)
        """
        # Get client history
        client_history = history.get_client_history(
            client_id,
            num_rounds=self.window_size
        )

        # Current update norm
        current_norm = self._compute_update_norm(update)

        if len(client_history) < 3:
            # Not enough history, use simple heuristic
            return 0.0

        # Compute baseline from history
        baseline = self._compute_baseline(client_history)

        # Compute anomaly score as z-score
        if baseline[1] > 0:  # std > 0
            score = abs(current_norm - baseline[0]) / baseline[1]
        else:
            score = 0.0

        return score

    def _compute_update_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of update."""
        total_norm = 0.0
        for param in update.values():
            total_norm += param.norm().item() ** 2
        return total_norm ** 0.5

    def _compute_baseline(
        self,
        client_history: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        Compute baseline (mean, std) from client history.

        Args:
            client_history: List of client behavior records

        Returns:
            (mean, std)
        """
        update_norms = [record.get("update_norm", 0.0) for record in client_history]

        if self.baseline_method == "moving_average":
            mean = np.mean(update_norms)
            std = np.std(update_norms)

        elif self.baseline_method == "exponential":
            # Exponential moving average
            weights = np.array([self.alpha ** (len(update_norms) - i - 1)
                               for i in range(len(update_norms))])
            weights = weights / weights.sum()
            mean = np.average(update_norms, weights=weights)
            std = np.sqrt(np.average((update_norms - mean) ** 2, weights=weights))

        elif self.baseline_method == "percentile":
            mean = np.percentile(update_norms, 50)  # Median
            std = np.percentile(update_norms, self.percentile) - mean

        else:
            mean = np.mean(update_norms)
            std = np.std(update_norms)

        return mean, std

    def _get_threshold(
        self,
        client_id: int,
        history: DefenseHistory,
    ) -> float:
        """Get threshold for anomaly detection."""
        if self.threshold_method == "static":
            return self.static_threshold
        else:
            # Adaptive threshold based on global detection rate
            avg_fp_rate = history.get_avg_fp_rate(num_rounds=10)

            # If FP rate is high, increase threshold
            if avg_fp_rate > 0.1:
                return self.static_threshold * 1.5
            else:
                return self.static_threshold

    def adapt_to_attack(
        self,
        attack_detected: bool,
        attack_type: str,
        history: DefenseHistory,
    ) -> None:
        """
        Adapt defense based on attack feedback.

        If attack detected, increase sensitivity.
        If FP rate high, decrease sensitivity.
        """
        avg_fp_rate = history.get_avg_fp_rate(num_rounds=10)

        if attack_detected and avg_fp_rate < 0.1:
            # Increase sensitivity
            self.current_threshold *= 0.9
            logger.info(
                f"Attack detected! Decreasing threshold to {self.current_threshold:.2f}"
            )
        elif avg_fp_rate > 0.15:
            # Too many false positives, decrease sensitivity
            self.current_threshold *= 1.1
            logger.info(
                f"High FP rate! Increasing threshold to {self.current_threshold:.2f}"
            )
