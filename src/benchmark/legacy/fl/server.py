"""Flower server for federated learning."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.server import ServerConfig, Server
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class FedPhishServer:
    """Flower server for phishing detection."""

    def __init__(
        self,
        strategy: Strategy,
        num_rounds: int,
        config: Optional[Dict] = None
    ):
        """
        Initialize server.

        Args:
            strategy: FL strategy
            num_rounds: Number of training rounds
            config: Server configuration
        """
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.config = config or {}

        # Server address
        self.address = self.config.get("address", "127.0.0.1:8080")

        # History tracking
        self.history = {
            "loss": [],
            "accuracy": [],
            "round_metrics": []
        }

    def start(self) -> Dict:
        """
        Start the FL server.

        Returns:
            Training history
        """
        # Create server configuration
        server_config = ServerConfig(num_rounds=self.num_rounds)

        # Start server (blocking)
        fl.server.start_server(
            server_address=self.address,
            config=server_config,
            strategy=self.strategy,
        )

        return self.history

    def evaluate_round(
        self,
        round_num: int,
        parameters: fl.common.Parameters
    ) -> Dict[str, float]:
        """
        Evaluate model at a round.

        Args:
            round_num: Round number
            parameters: Model parameters

        Returns:
            Evaluation metrics
        """
        # This would be called by the strategy during evaluation
        pass


def start_server(
    strategy: Strategy,
    num_rounds: int,
    address: str = "127.0.0.1:8080",
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to start server.

    Args:
        strategy: FL strategy
        num_rounds: Number of rounds
        address: Server address
        config: Additional configuration

    Returns:
        Training history
    """
    server = FedPhishServer(
        strategy=strategy,
        num_rounds=num_rounds,
        config={"address": address, **(config or {})}
    )
    return server.start()
