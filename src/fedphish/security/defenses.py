"""
Defense mechanisms against Byzantine attacks in federated learning.

Implements robust aggregation strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class ByzantineDefense:
    """Base class for Byzantine defenses."""

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
    ):
        """
        Initialize defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
        """
        self.num_clients = num_clients
        self.num_malicious = num_malicious or max(1, num_clients // 5)

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Apply defense and return aggregated result.

        Args:
            gradients: List of gradients from clients
            weights: Optional client weights

        Returns:
            Aggregated gradient
        """
        raise NotImplementedError


class KrumDefense(ByzantineDefense):
    """
    Krum defense strategy.

    Selects gradient closest to all others (robust to Byzantine).
    """

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
        multi_krum: bool = False,
    ):
        """
        Initialize Krum defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            multi_krum: Use Multi-Krum (select top-f closest)
        """
        super().__init__(num_clients, num_malicious)
        self.multi_krum = multi_krum

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Apply Krum defense."""
        # Flatten gradients for distance computation
        flat_grads = [g.flatten() for g in gradients]
        num_clients = len(gradients)

        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(flat_grads[i] - flat_grads[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores for each client
        # Sum of num_clients - num_malicious - 2 smallest distances
        scores = []
        for i in range(num_clients):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[:num_clients - self.num_malicious - 2])
            scores.append(score)

        # Select client with minimum score
        if self.multi_krum:
            # Multi-Krum: select multiple and average
            num_to_select = num_clients - self.num_malicious
            top_indices = np.argsort(scores)[:num_to_select]
            selected_grads = [gradients[i] for i in top_indices]
            return np.mean(selected_grads, axis=0)
        else:
            # Standard Krum: select single best
            best_idx = np.argmin(scores)
            return gradients[best_idx]


class FoolsGoldDefense(ByzantineDefense):
    """
    FoolsGold defense strategy.

    Detects and downweights malicious clients based on similarity.
    """

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
        learning_rate: float = 0.1,
    ):
        """
        Initialize FoolsGold defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            learning_rate: Learning rate for weight updates
        """
        super().__init__(num_clients, num_malicious)
        self.learning_rate = learning_rate
        self.client_history = []

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Apply FoolsGold defense."""
        # Flatten gradients
        flat_grads = np.array([g.flatten() for g in gradients])

        # Store history
        self.client_history.append(flat_grads)

        # Limit history size
        if len(self.client_history) > 10:
            self.client_history.pop(0)

        # Compute similarity matrix
        similarity = self._compute_similarity()

        # Compute FoolsGold weights
        fg_weights = self._compute_foolsgold_weights(similarity)

        # Weighted average
        weighted_grads = [
            w * g for w, g in zip(fg_weights, gradients)
        ]
        aggregated = np.sum(weighted_grads, axis=0)

        logger.debug(
            f"FoolsGold weights: min={fg_weights.min():.4f}, "
            f"max={fg_weights.max():.4f}"
        )

        return aggregated

    def _compute_similarity(self) -> np.ndarray:
        """Compute cosine similarity between clients."""
        # Stack history
        history = np.stack(self.client_history, axis=1)  # (num_clients, history_len, dim)

        # Compute mean for each client
        client_means = history.mean(axis=1)  # (num_clients, dim)

        # Compute cosine similarity
        similarity = np.zeros((self.num_clients, self.num_clients))
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                sim = np.dot(client_means[i], client_means[j]) / (
                    np.linalg.norm(client_means[i]) * np.linalg.norm(client_means[j]) + 1e-10
                )
                similarity[i, j] = sim
                similarity[j, i] = sim

        return similarity

    def _compute_foolsgold_weights(
        self,
        similarity: np.ndarray,
    ) -> np.ndarray:
        """Compute FoolsGold weights from similarity."""
        # Compute maximum similarity for each client
        max_similarities = np.max(similarity, axis=1)

        # Convert to weights (higher similarity = lower weight)
        # Malicious clients have high similarity with each other
        weights = 1.0 / (1.0 + max_similarities)

        # Normalize
        weights = weights / weights.sum()

        return weights


class TrimmedMeanDefense(ByzantineDefense):
    """
    Trimmed mean defense.

    Removes extreme values before averaging.
    """

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
        trim_fraction: float = 0.2,
    ):
        """
        Initialize trimmed mean defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            trim_fraction: Fraction to trim from each end
        """
        super().__init__(num_clients, num_malicious)
        self.trim_fraction = trim_fraction

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Apply trimmed mean defense."""
        # Compute norms for ordering
        norms = np.array([np.linalg.norm(g) for g in gradients])

        # Sort by norms
        sorted_indices = np.argsort(norms)

        # Trim extremes
        num_to_trim = int(len(gradients) * self.trim_fraction)
        trimmed_indices = sorted_indices[num_to_trim:len(gradients) - num_to_trim]

        # Average remaining
        trimmed_grads = [gradients[i] for i in trimmed_indices]
        aggregated = np.mean(trimmed_grads, axis=0)

        logger.debug(
            f"Trimmed mean: removed {2 * num_to_trim} extreme gradients"
        )

        return aggregated


class NormClippingDefense(ByzantineDefense):
    """
    Gradient norm clipping defense.

    Clips gradients with excessive norms.
    """

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
        max_norm: float = 1.0,
        median_multiplier: float = 3.0,
    ):
        """
        Initialize norm clipping defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            max_norm: Maximum absolute norm
            median_multiplier: Multiplier for median-based adaptive clipping
        """
        super().__init__(num_clients, num_malicious)
        self.max_norm = max_norm
        self.median_multiplier = median_multiplier

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Apply norm clipping defense."""
        # Compute norms
        norms = np.array([np.linalg.norm(g) for g in gradients])

        # Compute adaptive threshold based on median
        median_norm = np.median(norms)
        threshold = min(self.max_norm, self.median_multiplier * median_norm)

        # Clip gradients
        clipped_grads = []
        for grad, norm in zip(gradients, norms):
            if norm > threshold:
                clipped = grad * (threshold / norm)
                clipped_grads.append(clipped)
            else:
                clipped_grads.append(grad)

        # Average
        aggregated = np.mean(clipped_grads, axis=0)

        logger.debug(
            f"Norm clipping: median={median_norm:.4f}, "
            f"threshold={threshold:.4f}, clipped={(norms > threshold).sum()}"
        )

        return aggregated


class CombinedDefense(ByzantineDefense):
    """
    Combined defense using multiple strategies.

    Applies multiple defenses in sequence for robustness.
    """

    def __init__(
        self,
        num_clients: int,
        num_malicious: Optional[int] = None,
        defense_strategies: Optional[List[str]] = None,
    ):
        """
        Initialize combined defense.

        Args:
            num_clients: Total number of clients
            num_malicious: Estimated number of malicious clients
            defense_strategies: List of defense names to combine
        """
        super().__init__(num_clients, num_malicious)

        self.defense_strategies = defense_strategies or [
            "norm_clipping",
            "foolsgold",
        ]

        # Initialize individual defenses
        self.defenses = self._create_defenses()

        logger.info(
            f"Initialized combined defense with strategies: "
            f"{self.defense_strategies}"
        )

    def _create_defenses(self) -> List[ByzantineDefense]:
        """Create individual defense instances."""
        defenses = []

        for strategy in self.defense_strategies:
            if strategy == "krum":
                defenses.append(KrumDefense(self.num_clients, self.num_malicious))
            elif strategy == "foolsgold":
                defenses.append(FoolsGoldDefense(self.num_clients, self.num_malicious))
            elif strategy == "trimmed_mean":
                defenses.append(TrimmedMeanDefense(self.num_clients, self.num_malicious))
            elif strategy == "norm_clipping":
                defenses.append(NormClippingDefense(self.num_clients, self.num_malicious))
            else:
                logger.warning(f"Unknown defense strategy: {strategy}")

        return defenses

    def defend(
        self,
        gradients: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Apply combined defense."""
        current_grads = gradients

        # Apply each defense in sequence
        for defense in self.defenses:
            # Most defenses return aggregated gradient
            # For chained application, we need to handle differently
            if isinstance(defense, (FoolsGoldDefense, NormClippingDefense)):
                # These modify gradients internally
                result = defense.defend(current_grads, weights)
                # For next iteration, create list of same result
                current_grads = [result] * len(current_grads)
            else:
                # Krum and trimmed mean return aggregated result directly
                result = defense.defend(current_grads, weights)
                current_grads = [result] * len(current_grads)

        return result


class DefenseEvaluator:
    """
    Evaluate defense effectiveness against attacks.
    """

    def __init__(
        self,
        defense: ByzantineDefense,
    ):
        """
        Initialize evaluator.

        Args:
            defense: Defense to evaluate
        """
        self.defense = defense

    def evaluate(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Evaluate defense against attack.

        Args:
            gradients: Attacked gradients
            malicious_indices: Indices of malicious clients

        Returns:
            Evaluation metrics
        """
        # Apply defense
        aggregated = self.defense.defend(gradients)

        # Compute metrics
        metrics = {
            "defense_type": type(self.defense).__name__,
            "num_malicious": len(malicious_indices),
            "num_benign": len(gradients) - len(malicious_indices),
        }

        # Analyze robustness
        metrics["robustness_score"] = self._compute_robustness_score(
            gradients, malicious_indices, aggregated
        )

        return metrics

    def _compute_robustness_score(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
        aggregated: np.ndarray,
    ) -> float:
        """Compute robustness score."""
        # Compare aggregated with benign gradients
        benign_grads = [g for i, g in enumerate(gradients) if i not in malicious_indices]

        if not benign_grads:
            return 0.0

        # Compute distance from aggregated to benign mean
        benign_mean = np.mean(benign_grads, axis=0)
        distance = np.linalg.norm(aggregated - benign_mean)

        # Lower distance = higher robustness
        # Normalize by benign gradient norm
        benign_norm = np.linalg.norm(benign_mean)
        robustness = max(0, 1 - distance / (benign_norm + 1e-10))

        return robustness


def create_defense(
    defense_type: str,
    num_clients: int,
    num_malicious: Optional[int] = None,
    **kwargs,
) -> ByzantineDefense:
    """
    Create defense instance.

    Args:
        defense_type: Type of defense
        num_clients: Total number of clients
        num_malicious: Estimated number of malicious clients
        **kwargs: Defense-specific parameters

    Returns:
        Defense instance
    """
    if defense_type == "krum":
        return KrumDefense(num_clients, num_malicious, **kwargs)
    elif defense_type == "foolsgold":
        return FoolsGoldDefense(num_clients, num_malicious, **kwargs)
    elif defense_type == "trimmed_mean":
        return TrimmedMeanDefense(num_clients, num_malicious, **kwargs)
    elif defense_type == "norm_clipping":
        return NormClippingDefense(num_clients, num_malicious, **kwargs)
    elif defense_type == "combined":
        return CombinedDefense(num_clients, num_malicious, **kwargs)
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")
