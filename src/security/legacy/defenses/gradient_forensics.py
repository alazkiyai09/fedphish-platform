"""Gradient forensics defense implementation."""

import logging
from typing import Any, Dict, List, Tuple, Set

import numpy as np
from sklearn.cluster import DBSCAN
import torch

from .base import (
    BaseDefense,
    DefenseConfig,
    DefenseHistory,
    DefenderObservability,
)

logger = logging.getLogger(__name__)


class GradientForensics(BaseDefense):
    """
    Gradient forensics defense.

    Analyzes gradient structure beyond norms to detect
    coordinated attacks and identify attack types.
    """

    def __init__(
        self,
        defender_observability: DefenderObservability,
        analysis_method: str = "pca",
        coordination_threshold: float = 0.9,
        pca_variance_threshold: float = 0.95,
        clustering_method: str = "dbscan",
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 2,
    ):
        """
        Initialize gradient forensics defense.

        Args:
            defender_observability: Defender observability
            analysis_method: Analysis method ("pca", "clustering", "distance", "combined")
            coordination_threshold: Threshold for detecting coordination
            pca_variance_threshold: Variance threshold for PCA
            clustering_method: Clustering method ("dbscan", "kmeans", "hierarchical")
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN min_samples parameter
        """
        config = DefenseConfig(
            defense_type="gradient_forensics",
            analysis_method=analysis_method,
        )
        super().__init__(defender_observability, config)

        self.analysis_method = analysis_method
        self.coordination_threshold = coordination_threshold
        self.pca_variance_threshold = pca_variance_threshold
        self.clustering_method = clustering_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def detect(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
        round_num: int,
        history: DefenseHistory,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Detect malicious clients using gradient forensics.

        Args:
            client_updates: List of (client_id, parameters) tuples
            round_num: Current round
            history: Defense history

        Returns:
            (malicious_client_ids, detection_metadata)
        """
        malicious_ids = []

        # Detect coordinated groups
        coordinated_groups = self.detect_coordination([
            params for _, params in client_updates
        ])

        # Identify which groups are malicious
        for group in coordinated_groups:
            if len(group) >= 2:  # Group of 2+ suggests coordination
                # Check if group behavior is anomalous
                if self._is_group_anomalous(group, client_updates):
                    malicious_ids.extend(list(group))

        # Classify attack type
        attack_type = self.classify_attack_type([
            params for _, params in client_updates
        ])

        metadata = {
            "coordinated_groups": [list(g) for g in coordinated_groups],
            "num_coordinated_groups": len(coordinated_groups),
            "attack_type": attack_type,
            "analysis_method": self.analysis_method,
        }

        logger.debug(
            f"Gradient forensics: detected {len(malicious_ids)} malicious clients, "
            f"attack type: {attack_type}"
        )

        return malicious_ids, metadata

    def detect_coordination(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> List[Set[int]]:
        """
        Detect groups of coordinated clients (Sybil attacks).

        Args:
            updates: List of client updates

        Returns:
            List of sets containing coordinated client indices
        """
        if self.analysis_method in ["pca", "combined"]:
            return self._detect_coordination_pca(updates)
        elif self.analysis_method in ["clustering", "combined"]:
            return self._detect_coordination_clustering(updates)
        else:  # distance
            return self._detect_coordination_distance(updates)

    def _detect_coordination_pca(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> List[Set[int]]:
        """Detect coordination using PCA."""
        if len(updates) < 2:
            return []

        # Flatten updates
        flattened = []
        for update in updates:
            flat = torch.cat([param.flatten() for param in update.values()])
            flattened.append(flat.numpy())

        flattened = np.array(flattened)

        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, len(updates)))
        transformed = pca.fit_transform(flattened)

        # Find similar pairs in PCA space
        coordinated_groups = []
        for i in range(len(transformed)):
            for j in range(i + 1, len(transformed)):
                # Compute cosine similarity in PCA space
                similarity = np.dot(
                    transformed[i],
                    transformed[j]
                ) / (
                    np.linalg.norm(transformed[i]) * np.linalg.norm(transformed[j]) + 1e-8
                )

                if similarity > self.coordination_threshold:
                    # Check if already in a group
                    found = False
                    for group in coordinated_groups:
                        if i in group or j in group:
                            group.add(i)
                            group.add(j)
                            found = True
                            break

                    if not found:
                        coordinated_groups.append({i, j})

        return coordinated_groups

    def _detect_coordination_clustering(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> List[Set[int]]:
        """Detect coordination using clustering."""
        if len(updates) < self.dbscan_min_samples:
            return []

        # Flatten updates
        flattened = []
        for update in updates:
            flat = torch.cat([param.flatten() for param in update.values()])
            flattened.append(flat.numpy())

        flattened = np.array(flattened)

        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        )
        labels = clustering.fit_predict(flattened)

        # Group by cluster
        from collections import defaultdict
        clusters = defaultdict(set)
        for idx, label in enumerate(labels):
            if label != -1:  # Not noise
                clusters[label].add(idx)

        # Return groups with 2+ members
        return [group for group in clusters.values() if len(group) >= 2]

    def _detect_coordination_distance(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> List[Set[int]]:
        """Detect coordination using distance-based method."""
        if len(updates) < 2:
            return []

        # Flatten updates
        flattened = []
        for update in updates:
            flat = torch.cat([param.flatten() for param in update.values()])
            flattened.append(flat)

        # Find similar pairs
        coordinated_groups = []
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    flattened[i].unsqueeze(0),
                    flattened[j].unsqueeze(0)
                ).item()

                if similarity > self.coordination_threshold:
                    # Check if already in a group
                    found = False
                    for group in coordinated_groups:
                        if i in group or j in group:
                            group.add(i)
                            group.add(j)
                            found = True
                            break

                    if not found:
                        coordinated_groups.append({i, j})

        return coordinated_groups

    def _is_group_anomalous(
        self,
        group: Set[int],
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> bool:
        """Check if a coordinated group is anomalous (malicious)."""
        # Get group updates
        group_updates = [
            params for idx, (_, params) in enumerate(client_updates)
            if idx in group
        ]

        # Compute group centroid
        group_centroid = self._compute_centroid(group_updates)

        # Compute distance from group centroid to overall centroid
        all_updates = [params for _, params in client_updates]
        overall_centroid = self._compute_centroid(all_updates)

        distance = self._compute_distance(group_centroid, overall_centroid)

        # If far from overall centroid, likely malicious
        return distance > self.coordination_threshold

    def _compute_centroid(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute centroid of updates."""
        if not updates:
            return {}

        centroid = {}
        for name in updates[0].keys():
            stacked = torch.stack([u[name] for u in updates])
            centroid[name] = torch.mean(stacked, dim=0)

        return centroid

    def _compute_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """Compute cosine distance between updates."""
        flat1 = torch.cat([param.flatten() for param in update1.values()])
        flat2 = torch.cat([param.flatten() for param in update2.values()])

        similarity = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0),
            flat2.unsqueeze(0)
        ).item()

        return 1.0 - similarity

    def classify_attack_type(
        self,
        gradients: List[Dict[str, torch.Tensor]],
    ) -> str:
        """
        Identify attack type from gradient signature.

        Args:
            gradients: List of gradients

        Returns:
            Attack type string
        """
        if len(gradients) < 2:
            return "unknown"

        # Analyze gradient patterns
        norms = []
        for grad in gradients:
            flat = torch.cat([param.flatten() for param in grad.values()])
            norms.append(flat.norm().item())

        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        # Check for label flip (high variance)
        if std_norm > mean_norm * 0.5:
            return "label_flip"

        # Check for model poisoning (very high norms)
        if mean_norm > 10.0:
            return "model_poisoning"

        # Check for backdoor (consistent direction)
        if std_norm < mean_norm * 0.1:
            return "backdoor"

        return "unknown"

    def adapt_to_attack(
        self,
        attack_detected: bool,
        attack_type: str,
        history: DefenseHistory,
    ) -> None:
        """
        Adapt defense based on attack feedback.

        If attack detected, adjust coordination threshold.
        """
        avg_fp_rate = history.get_avg_fp_rate(num_rounds=10)

        if attack_detected and avg_fp_rate < 0.1:
            # Increase sensitivity (decrease threshold)
            self.coordination_threshold = max(
                0.7,
                self.coordination_threshold * 0.95
            )
            logger.info(
                f"Attack detected! Decreasing coordination threshold to "
                f"{self.coordination_threshold:.2f}"
            )
        elif avg_fp_rate > 0.15:
            # Too many false positives, increase threshold
            self.coordination_threshold = min(
                0.99,
                self.coordination_threshold * 1.05
            )
            logger.info(
                f"High FP rate! Increasing coordination threshold to "
                f"{self.coordination_threshold:.2f}"
            )
