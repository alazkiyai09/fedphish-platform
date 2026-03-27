"""FL strategies: FedAvg, FedProx."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class FedAvgStrategy(FedAvg):
    """FedAvg strategy for phishing detection."""

    def __init__(
        self,
        *,
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.8,
        min_fit_clients: int = 8,
        min_evaluate_clients: int = 8,
        min_available_clients: int = 10,
        **kwargs
    ):
        """
        Initialize FedAvg strategy.

        Args:
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum available clients
            **kwargs: Additional arguments
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )


class FedProxStrategy(FedAvg):
    """FedProx strategy with proximal term."""

    def __init__(
        self,
        mu: float = 0.01,
        **kwargs
    ):
        """
        Initialize FedProx strategy.

        Args:
            mu: Proximal term coefficient
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.mu = mu

    def aggregate_fit(
        self,
        server_round: int,
        results: list,
        failures: list
    ):
        """
        Aggregate fit results using FedAvg (proximal term is handled by clients).

        Args:
            server_round: Current round
            results: List of (client, fit result) tuples
            failures: List of failures

        Returns:
            Aggregated parameters
        """
        # FedProx uses standard FedAvg aggregation
        # The proximal term is added to client-side loss
        return super().aggregate_fit(server_round, results, failures)


def get_strategy(
    strategy_name: str,
    config: Optional[Dict] = None
):
    """
    Get FL strategy by name.

    Args:
        strategy_name: Name of strategy (fedavg, fedprox)
        config: Strategy configuration

    Returns:
        Strategy instance
    """
    config = config or {}

    if strategy_name == "fedavg":
        return FedAvgStrategy(
            fraction_fit=config.get("fraction_fit", 0.8),
            fraction_evaluate=config.get("fraction_evaluate", 0.8),
            min_fit_clients=config.get("min_fit_clients", 8),
            min_evaluate_clients=config.get("min_evaluate_clients", 8),
            min_available_clients=config.get("min_available_clients", 10),
        )
    elif strategy_name == "fedprox":
        return FedProxStrategy(
            mu=config.get("mu", 0.01),
            fraction_fit=config.get("fraction_fit", 0.8),
            fraction_evaluate=config.get("fraction_evaluate", 0.8),
            min_fit_clients=config.get("min_fit_clients", 8),
            min_evaluate_clients=config.get("min_evaluate_clients", 8),
            min_available_clients=config.get("min_available_clients", 10),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
