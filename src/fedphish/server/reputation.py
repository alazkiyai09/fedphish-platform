"""
Reputation system for clients.

Tracks client reliability and adjusts aggregation weights.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BankReputation:
    """
    Reputation system for federated learning clients (banks).

    Tracks per-bank reliability based on:
    - Proof validity
    - Update similarity
    - Historical contribution
    """

    def __init__(
        self,
        num_clients: int,
        initial_reputation: float = 1.0,
        alpha: float = 0.1,  # Exponential moving average factor
    ):
        """
        Initialize reputation system.

        Args:
            num_clients: Number of clients
            initial_reputation: Starting reputation for all clients
            alpha: EMA factor for updating reputation
        """
        self.num_clients = num_clients
        self.initial_reputation = initial_reputation
        self.alpha = alpha

        # Reputation scores
        self.reputations = np.ones(num_clients) * initial_reputation

        # History
        self.update_history = {}
        for i in range(num_clients):
            self.update_history[i] = []

        self.round_count = 0

        logger.info(
            f"Initialized reputation system: num_clients={num_clients}, "
            f"alpha={alpha}"
        )

    def update_reputation(
        self,
        client_id: int,
        proof_valid: bool,
        update_similarity: float,
        contribution_score: float = 1.0,
    ):
        """
        Update reputation for a client.

        Args:
            client_id: Client ID
            proof_valid: Whether ZK proof was valid
            update_similarity: Similarity to aggregated update
            contribution_score: Contribution score (e.g., data size)
        """
        # Compute new reputation component
        if proof_valid:
            proof_score = 1.0
        else:
            proof_score = 0.0

        # Combine factors
        new_reputation_component = (
            0.4 * proof_score +
            0.4 * update_similarity +
            0.2 * contribution_score
        )

        # Update with exponential moving average
        current = self.reputations[client_id]
        updated = (1 - self.alpha) * current + self.alpha * new_reputation_component

        # Ensure in [0, 1]
        self.reputations[client_id] = np.clip(updated, 0.0, 1.0)

        # Record history
        self.update_history[client_id].append({
            "round": self.round_count,
            "reputation": self.reputations[client_id],
            "proof_valid": proof_valid,
            "similarity": update_similarity,
        })

        logger.debug(
            f"Updated reputation for client {client_id}: "
            f"{current:.4f} -> {self.reputations[client_id]:.4f}"
        )

    def get_reputation(self, client_id: int) -> float:
        """Get reputation for a client."""
        return self.reputations[client_id]

    def get_all_reputations(self) -> np.ndarray:
        """Get all reputations."""
        return self.reputations.copy()

    def increment_round(self):
        """Increment round counter."""
        self.round_count += 1

    def get_reputation_weights(self) -> np.ndarray:
        """
        Get aggregation weights based on reputations.

        Returns:
            Normalized weights
        """
        # Ensure non-negative
        weights = np.maximum(self.reputations, 0.01)

        # Normalize
        weights = weights / weights.sum()

        return weights

    def get_reputation_summary(self) -> Dict[str, Any]:
        """Get reputation summary statistics."""
        return {
            "mean": float(np.mean(self.reputations)),
            "std": float(np.std(self.reputations)),
            "min": float(np.min(self.reputations)),
            "max": float(np.max(self.reputations)),
            "round": self.round_count,
            "clients": {
                i: float(self.reputations[i])
                for i in range(self.num_clients)
            },
        }


class ReputationScorer:
    """
    Score client updates for reputation system.

    Computes similarity and contribution scores.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize scorer.

        Args:
            similarity_threshold: Minimum similarity for good score
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity(
        self,
        update: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Compute similarity between update and reference.

        Args:
            update: Client update
            reference: Reference (e.g., aggregated update)

        Returns:
            Similarity score in [0, 1]
        """
        # Cosine similarity
        dot = np.dot(update.flatten(), reference.flatten())
        norm_update = np.linalg.norm(update)
        norm_reference = np.linalg.norm(reference)

        if norm_update == 0 or norm_reference == 0:
            return 0.0

        cosine_sim = dot / (norm_update * norm_reference)

        # Clip to [0, 1]
        cosine_sim = max(0, min(1, cosine_sim))

        return float(cosine_sim)

    def compute_contribution_score(
        self,
        num_samples: int,
        loss: float,
    ) -> float:
        """
        Compute contribution score.

        Args:
            num_samples: Number of training samples
            loss: Training loss

        Returns:
            Contribution score
        """
        # More samples = higher contribution
        # Lower loss = higher contribution (model improved)

        # Normalize (assume max 10000 samples)
        sample_score = min(1.0, num_samples / 10000.0)

        # Loss score (lower is better, assume max loss 5.0)
        loss_score = max(0.0, 1.0 - loss / 5.0)

        # Combine
        contribution = 0.5 * sample_score + 0.5 * loss_score

        return float(contribution)

    def score_update(
        self,
        client_id: int,
        update: np.ndarray,
        reference: np.ndarray,
        num_samples: int,
        loss: float,
    ) -> Dict[str, float]:
        """
        Compute all scores for an update.

        Args:
            client_id: Client ID
            update: Client update
            reference: Reference update
            num_samples: Number of samples
            loss: Training loss

        Returns:
            Dictionary of scores
        """
        similarity = self.compute_similarity(update, reference)
        contribution = self.compute_contribution_score(num_samples, loss)

        return {
            "similarity": similarity,
            "contribution": contribution,
        }


class ReputationWeight:
    """
    Convert reputations to aggregation weights.
    """

    @staticmethod
    def reputation_to_weights(
        reputations: np.ndarray,
        weighting_scheme: str = "linear",  # 'linear', 'softmax', 'rank'
    ) -> np.ndarray:
        """
        Convert reputations to weights.

        Args:
            reputations: Reputation scores
            weighting_scheme: How to convert to weights

        Returns:
            Normalized weights
        """
        if weighting_scheme == "linear":
            # Direct proportional weighting
            weights = reputations / reputations.sum()

        elif weighting_scheme == "softmax":
            # Softmax weighting
            exp_reps = np.exp(reputations * 5)  # Temperature scaling
            weights = exp_reps / exp_reps.sum()

        elif weighting_scheme == "rank":
            # Rank-based weighting
            ranks = np.argsort(np.argsort(-reputations))  # Descending ranks
            # Linearly decreasing weights by rank
            weights = len(reputations) - ranks
            weights = weights / weights.sum()

        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")

        return weights
