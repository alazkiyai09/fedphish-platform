"""Honeypot defense implementation."""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import (
    BaseDefense,
    DefenseConfig,
    DefenseHistory,
    DefenderObservability,
)

logger = logging.getLogger(__name__)


class HoneypotDefense(BaseDefense):
    """
    Honeypot defense.

    Injects known-good updates from simulated honest clients
    to detect attackers by deviation.
    """

    def __init__(
        self,
        defender_observability: DefenderObservability,
        num_honeypots: int = 3,
        honeypot_strategy: str = "random",
        deviation_threshold: float = 2.0,
        distance_metric: str = "cosine",
    ):
        """
        Initialize honeypot defense.

        Args:
            defender_observability: Defender observability
            num_honeypots: Number of honeypot clients
            honeypot_strategy: How to place honeypots ("random", "cluster_based")
            deviation_threshold: Threshold for detecting deviation
            distance_metric: Distance metric ("cosine", "euclidean", "manhattan")
        """
        config = DefenseConfig(
            defense_type="honeypot",
            num_honeypots=num_honeypots,
        )
        super().__init__(defender_observability, config)

        self.num_honeypots = num_honeypots
        self.honeypot_strategy = honeypot_strategy
        self.deviation_threshold = deviation_threshold
        self.distance_metric = distance_metric

        # Honeypot client IDs (will be assigned dynamically)
        self.honeypot_ids: List[int] = []

    def detect(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
        round_num: int,
        history: DefenseHistory,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Detect malicious clients using honeypot comparison.

        Args:
            client_updates: List of (client_id, parameters) tuples
            round_num: Current round
            history: Defense history

        Returns:
            (malicious_client_ids, detection_metadata)
        """
        if len(client_updates) <= self.num_honeypots:
            # Not enough clients to have honeypots
            return [], {"num_honeypots": 0}

        # Assign honeypot IDs if first round or changed
        if not self.honeypot_ids or len(self.honeypot_ids) != self.num_honeypots:
            self._assign_honeypot_ids(client_updates)

        # Get honeypot updates
        honeypot_updates = [
            (cid, params) for cid, params in client_updates
            if cid in self.honeypot_ids
        ]

        # Get non-honeypot updates
        client_updates_non_honeypot = [
            (cid, params) for cid, params in client_updates
            if cid not in self.honeypot_ids
        ]

        # Compute honeypot centroid
        honeypot_centroid = self._compute_centroid([
            params for _, params in honeypot_updates
        ])

        # Detect deviations
        malicious_ids = []
        deviations = {}

        for client_id, params in client_updates_non_honeypot:
            deviation = self._compute_distance(params, honeypot_centroid)
            deviations[client_id] = deviation

            if deviation > self.deviation_threshold:
                malicious_ids.append(client_id)

        metadata = {
            "honeypot_ids": self.honeypot_ids,
            "deviations": deviations,
            "num_malicious": len(malicious_ids),
            "num_honeypots": self.num_honeypots,
            "deviation_threshold": self.deviation_threshold,
        }

        logger.debug(
            f"Honeypot defense: detected {len(malicious_ids)} malicious clients "
            f"(using {len(self.honeypot_ids)} honeypots)"
        )

        return malicious_ids, metadata

    def _assign_honeypot_ids(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> None:
        """Assign honeypot client IDs."""
        all_client_ids = [cid for cid, _ in client_updates]

        if self.honeypot_strategy == "random":
            # Random selection
            import random
            self.honeypot_ids = random.sample(
                all_client_ids,
                min(self.num_honeypots, len(all_client_ids))
            )

        elif self.honeypot_strategy == "cluster_based":
            # Select clients that are closest to centroid (likely honest)
            centroid = self._compute_centroid([params for _, params in client_updates])

            distances = []
            for cid, params in client_updates:
                dist = self._compute_distance(params, centroid)
                distances.append((dist, cid))

            # Select closest to centroid
            distances.sort()
            self.honeypot_ids = [cid for _, cid in distances[:self.num_honeypots]]

    def _compute_centroid(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute centroid of updates."""
        if not updates:
            return {}

        centroid = {}
        for name in updates[0].keys():
            # Stack and average
            stacked = torch.stack([u[name] for u in updates])
            centroid[name] = torch.mean(stacked, dim=0)

        return centroid

    def _compute_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute distance between two updates."""
        if self.distance_metric == "cosine":
            return self._cosine_distance(update1, update2)
        elif self.distance_metric == "euclidean":
            return self._euclidean_distance(update1, update2)
        elif self.distance_metric == "manhattan":
            return self._manhattan_distance(update1, update2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _cosine_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute cosine distance."""
        # Flatten updates
        flat1 = torch.cat([param.flatten() for param in update1.values()])
        flat2 = torch.cat([param.flatten() for param in update2.values()])

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0),
            flat2.unsqueeze(0)
        ).item()

        # Convert to distance
        return 1.0 - similarity

    def _euclidean_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute Euclidean distance."""
        # Flatten updates
        flat1 = torch.cat([param.flatten() for param in update1.values()])
        flat2 = torch.cat([param.flatten() for param in update2.values()])

        return torch.norm(flat1 - flat2).item()

    def _manhattan_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute Manhattan distance."""
        # Flatten updates
        flat1 = torch.cat([param.flatten() for param in update1.values()])
        flat2 = torch.cat([param.flatten() for param in update2.values()])

        return torch.sum(torch.abs(flat1 - flat2)).item()

    def adapt_to_attack(
        self,
        attack_detected: bool,
        attack_type: str,
        history: DefenseHistory,
    ) -> None:
        """
        Adapt defense based on attack feedback.

        If attack detected, increase deviation threshold or add honeypots.
        """
        avg_fp_rate = history.get_avg_fp_rate(num_rounds=10)

        if attack_detected and avg_fp_rate < 0.1:
            # Increase threshold (become less strict)
            self.deviation_threshold *= 1.1
            logger.info(
                f"Attack detected! Increasing deviation threshold to "
                f"{self.deviation_threshold:.2f}"
            )
        elif avg_fp_rate > 0.15:
            # Too many false positives, increase threshold more
            self.deviation_threshold *= 1.2
            logger.info(
                f"High FP rate! Increasing deviation threshold to "
                f"{self.deviation_threshold:.2f}"
            )
        elif not attack_detected and avg_fp_rate < 0.05:
            # Very low FP rate, can decrease threshold
            self.deviation_threshold *= 0.95
            logger.debug(
                f"Decreasing deviation threshold to {self.deviation_threshold:.2f}"
            )
