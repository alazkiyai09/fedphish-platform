"""
Byzantine defense strategies for server.

Detects and mitigates malicious client updates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedphish.security.defenses import (
    ByzantineDefense,
    CombinedDefense,
    FoolsGoldDefense,
    KrumDefense,
    NormClippingDefense,
    TrimmedMeanDefense,
    create_defense,
)

logger = logging.getLogger(__name__)


class ByzantineDetector:
    """
    Detect Byzantine (malicious) clients.

    Identifies suspicious updates using various strategies.
    """

    def __init__(
        self,
        num_clients: int,
        estimated_malicious: int = None,
        defense_strategy: str = "foolsgold",
    ):
        """
        Initialize Byzantine detector.

        Args:
            num_clients: Total number of clients
            estimated_malicious: Estimated number of malicious clients
            defense_strategy: Defense strategy to use
        """
        self.num_clients = num_clients
        self.estimated_malicious = estimated_malicious or max(1, num_clients // 5)
        self.defense_strategy = defense_strategy

        # Create defense
        self.defense = create_defense(
            defense_strategy,
            num_clients,
            self.estimated_malicious,
        )

        self.detection_history = []

        logger.info(
            f"Initialized Byzantine detector: strategy={defense_strategy}, "
            f"estimated_malicious={self.estimated_malicious}"
        )

    def detect_and_aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Detect malicious clients and aggregate.

        Args:
            updates: Client updates
            weights: Optional client weights

        Returns:
            Tuple of (aggregated_update, detected_malicious_ids)
        """
        # Apply defense to get aggregated result
        aggregated = self.defense.defend(updates, weights)

        # Detect malicious (simplified - in practice, defense would identify them)
        # For now, use norm-based detection
        malicious_ids = self._detect_by_norm(updates)

        # Record detection
        self.detection_history.append({
            "num_clients": len(updates),
            "num_detected": len(malicious_ids),
            "malicious_ids": malicious_ids,
        })

        logger.info(
            f"Detected {len(malicious_ids)} potentially malicious clients"
        )

        return aggregated, malicious_ids

    def _detect_by_norm(
        self,
        updates: List[np.ndarray],
    ) -> List[int]:
        """Detect malicious clients by gradient norm."""
        norms = [np.linalg.norm(u) for u in updates]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        # Detect outliers (3 standard deviations from mean)
        threshold = mean_norm + 3 * std_norm
        malicious = [
            i for i, norm in enumerate(norms)
            if norm > threshold
        ]

        return malicious

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {}

        total_rounds = len(self.detection_history)
        total_detections = sum(h["num_detected"] for h in self.detection_history)

        return {
            "total_rounds": total_rounds,
            "total_detections": total_detections,
            "avg_detections_per_round": total_detections / total_rounds if total_rounds > 0 else 0,
        }


class AttackMitigator:
    """
    Mitigate attacks from detected malicious clients.

    Filters or weights down malicious updates.
    """

    def __init__(
        self,
        mitigation_strategy: str = "filter",  # 'filter', 'downweight', 'correct'
    ):
        """
        Initialize attack mitigator.

        Args:
            mitigation_strategy: Strategy for mitigation
        """
        self.mitigation_strategy = mitigation_strategy

        logger.info(f"Initialized attack mitigator: strategy={mitigation_strategy}")

    def mitigate(
        self,
        updates: List[np.ndarray],
        client_ids: List[int],
        malicious_ids: List[int],
        weights: Optional[List[float]] = None,
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """
        Mitigate malicious updates.

        Args:
            updates: All client updates
            client_ids: Client IDs
            malicious_ids: Detected malicious client IDs
            weights: Optional client weights

        Returns:
            Tuple of (filtered_updates, filtered_client_ids, filtered_weights)
        """
        if self.mitigation_strategy == "filter":
            return self._filter_malicious(updates, client_ids, malicious_ids, weights)
        elif self.mitigation_strategy == "downweight":
            return self._downweight_malicious(updates, client_ids, malicious_ids, weights)
        elif self.mitigation_strategy == "correct":
            return self._correct_malicious(updates, client_ids, malicious_ids, weights)
        else:
            logger.warning(f"Unknown mitigation strategy: {self.mitigation_strategy}")
            return updates, client_ids, weights or [1.0] * len(updates)

    def _filter_malicious(
        self,
        updates: List[np.ndarray],
        client_ids: List[int],
        malicious_ids: List[int],
        weights: Optional[List[float]],
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """Filter out malicious updates."""
        filtered_updates = []
        filtered_ids = []
        filtered_weights = []

        for update, cid, weight in zip(
            updates,
            client_ids,
            weights or [1.0] * len(updates)
        ):
            if cid not in malicious_ids:
                filtered_updates.append(update)
                filtered_ids.append(cid)
                filtered_weights.append(weight)

        logger.info(f"Filtered out {len(malicious_ids)} malicious updates")

        return filtered_updates, filtered_ids, filtered_weights

    def _downweight_malicious(
        self,
        updates: List[np.ndarray],
        client_ids: List[int],
        malicious_ids: List[int],
        weights: Optional[List[float]],
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """Downweight malicious updates."""
        if weights is None:
            weights = [1.0] * len(updates)

        downweighted = []
        for cid, weight in zip(client_ids, weights):
            if cid in malicious_ids:
                # Reduce weight significantly
                downweighted.append(weight * 0.1)
            else:
                downweighted.append(weight)

        logger.info(f"Downweighted {len(malicious_ids)} malicious updates")

        return updates, client_ids, downweighted

    def _correct_malicious(
        self,
        updates: List[np.ndarray],
        client_ids: List[int],
        malicious_ids: List[int],
        weights: Optional[List[float]],
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """Correct malicious updates (replace with median)."""
        # Compute median of benign updates
        benign_updates = [
            update for update, cid in zip(updates, client_ids)
            if cid not in malicious_ids
        ]

        if benign_updates:
            median_update = np.median(benign_updates, axis=0)

            # Replace malicious updates with median
            corrected_updates = []
            for update, cid in zip(updates, client_ids):
                if cid in malicious_ids:
                    corrected_updates.append(median_update)
                else:
                    corrected_updates.append(update)

            logger.info(f"Corrected {len(malicious_ids)} malicious updates")

            return corrected_updates, client_ids, weights or [1.0] * len(updates)
        else:
            # All are malicious, return as-is
            return updates, client_ids, weights or [1.0] * len(updates)


class AdaptiveDefense:
    """
    Adaptive defense that adjusts based on threat level.

    Changes defense strategy based on detected attack patterns.
    """

    def __init__(
        self,
        num_clients: int,
        initial_strategy: str = "foolsgold",
    ):
        """
        Initialize adaptive defense.

        Args:
            num_clients: Number of clients
            initial_strategy: Initial defense strategy
        """
        self.num_clients = num_clients
        self.current_strategy = initial_strategy
        self.attack_history = []

        # Defense strategies in order of strength
        self.strategy_hierarchy = [
            "norm_clipping",
            "trimmed_mean",
            "foolsgold",
            "krum",
            "combined",
        ]

    def update_defense(
        self,
        detection_rate: float,
    ) -> str:
        """
        Update defense strategy based on detection rate.

        Args:
            detection_rate: Fraction of clients detected as malicious

        Returns:
            New defense strategy
        """
        self.attack_history.append(detection_rate)

        # Adjust strategy based on threat level
        if detection_rate > 0.3:
            # High threat - use strongest defense
            new_strategy = "combined"
        elif detection_rate > 0.1:
            # Medium threat - use Krum
            new_strategy = "krum"
        elif detection_rate > 0.05:
            # Low threat - use FoolsGold
            new_strategy = "foolsgold"
        else:
            # Minimal threat - use norm clipping
            new_strategy = "norm_clipping"

        if new_strategy != self.current_strategy:
            logger.info(
                f"Adaptive defense: {self.current_strategy} -> {new_strategy} "
                f"(detection_rate={detection_rate:.3f})"
            )
            self.current_strategy = new_strategy

        return self.current_strategy

    def get_current_defense(self) -> ByzantineDefense:
        """Get current defense instance."""
        return create_defense(
            self.current_strategy,
            self.num_clients,
        )
