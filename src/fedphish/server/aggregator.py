"""
Aggregation strategies for federated learning server.

Implements HT2ML hybrid aggregation and weighted averaging.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedphish.privacy.ht2ml import HT2MLAggregator, PrivacyLevel
from fedphish.security.defenses import ByzantineDefense

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Base aggregator for federated learning.
    """

    def __init__(
        self,
        num_clients: int,
        aggregation_strategy: str = "weighted_average",
    ):
        """
        Initialize aggregator.

        Args:
            num_clients: Number of clients
            aggregation_strategy: Strategy name
        """
        self.num_clients = num_clients
        self.aggregation_strategy = aggregation_strategy

    def aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Aggregate updates."""
        raise NotImplementedError


class HT2MLAggregatorServer(FederatedAggregator):
    """
    HT2ML hybrid aggregator.

    Combines HE and TEE for privacy-preserving aggregation.
    """

    def __init__(
        self,
        num_clients: int,
        privacy_level: PrivacyLevel = PrivacyLevel.LEVEL_3,
        use_real_tee: bool = False,
    ):
        """
        Initialize HT2ML aggregator.

        Args:
            num_clients: Number of clients
            privacy_level: Privacy level
            use_real_tee: Whether to use real TEE
        """
        super().__init__(num_clients, "ht2ml")

        self.aggregator = HT2MLAggregator(
            privacy_level=privacy_level,
            use_real_tee=use_real_tee,
        )

        logger.info(f"Initialized HT2ML aggregator at level {privacy_level.name}")

    def aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
        operation: str = "average",
    ) -> np.ndarray:
        """
        Aggregate updates using HT2ML.

        Args:
            updates: Client updates
            weights: Optional weights
            operation: Aggregation operation

        Returns:
            Aggregated result
        """
        result = self.aggregator.aggregate(
            updates=updates,
            weights=weights,
            operation=operation,
        )

        logger.debug(f"Aggregated {len(updates)} updates using HT2ML")

        return result


class WeightedAggregator(FederatedAggregator):
    """
    Weighted averaging aggregator.

    Uses client weights (e.g., based on data size or reputation).
    """

    def __init__(
        self,
        num_clients: int,
        default_weights: Optional[List[float]] = None,
    ):
        """
        Initialize weighted aggregator.

        Args:
            num_clients: Number of clients
            default_weights: Default weights (uniform if None)
        """
        super().__init__(num_clients, "weighted_average")

        if default_weights is None:
            self.default_weights = np.ones(num_clients) / num_clients
        else:
            self.default_weights = np.array(default_weights)
            self.default_weights = self.default_weights / self.default_weights.sum()

    def aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Aggregate with weighted average."""
        if weights is None:
            weights = self.default_weights

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average
        aggregated = sum(w * u for w, u in zip(weights, updates))

        logger.debug(f"Aggregated {len(updates)} updates with weighted average")

        return aggregated


class ClippingAggregator(FederatedAggregator):
    """
    Gradient norm clipping aggregator.

    Clips gradients before aggregation.
    """

    def __init__(
        self,
        num_clients: int,
        max_norm: float = 1.0,
        base_aggregator: Optional[FederatedAggregator] = None,
    ):
        """
        Initialize clipping aggregator.

        Args:
            num_clients: Number of clients
            max_norm: Maximum gradient norm
            base_aggregator: Base aggregator (weighted if None)
        """
        super().__init__(num_clients, "clipping")

        self.max_norm = max_norm
        self.base_aggregator = base_aggregator or WeightedAggregator(num_clients)

    def aggregate(
        self,
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Aggregate with norm clipping."""
        # Clip gradients
        clipped_updates = []
        for update in updates:
            norm = np.linalg.norm(update)
            if norm > self.max_norm:
                clipped = update * (self.max_norm / norm)
            else:
                clipped = update
            clipped_updates.append(clipped)

        # Aggregate with base aggregator
        result = self.base_aggregator.aggregate(clipped_updates, weights)

        logger.debug(f"Aggregated with clipping, max_norm={self.max_norm}")

        return result


class AveragingStrategy:
    """Factory for creating aggregation strategies."""

    @staticmethod
    def create(
        strategy: str,
        num_clients: int,
        **kwargs,
    ) -> FederatedAggregator:
        """
        Create aggregator instance.

        Args:
            strategy: Strategy name
            num_clients: Number of clients
            **kwargs: Strategy-specific parameters

        Returns:
            Aggregator instance
        """
        if strategy == "fedavg" or strategy == "weighted_average":
            return WeightedAggregator(num_clients, **kwargs)
        elif strategy == "ht2ml":
            return HT2MLAggregatorServer(num_clients, **kwargs)
        elif strategy == "clipping":
            return ClippingAggregator(num_clients, **kwargs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")


def create_aggregator(
    strategy: str,
    num_clients: int,
    **kwargs,
) -> FederatedAggregator:
    """
    Create aggregator instance.

    Args:
        strategy: Strategy name
        num_clients: Number of clients
        **kwargs: Additional parameters

    Returns:
        Aggregator instance
    """
    return AveragingStrategy.create(strategy, num_clients, **kwargs)
