"""Federated Learning server."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .aggregation import FedAvg
from ..defenses.base import BaseDefense

logger = logging.getLogger(__name__)


class FLServer:
    """Federated Learning server."""

    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        learning_rate: float = 1.0,
        aggregation_strategy: str = "fedavg",
        defense: Optional[BaseDefense] = None,
        device: str = "cpu",
    ):
        """
        Initialize FL server.

        Args:
            model: Global model
            num_clients: Total number of clients
            learning_rate: Server learning rate for aggregation
            aggregation_strategy: Aggregation method ("fedavg", "krum", "multi_krum", "trimmed_mean")
            defense: Defense mechanism
            device: Device to use
        """
        self.model = model.to(device)
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.aggregation_strategy = aggregation_strategy
        self.defense = defense
        self.device = device

        # History tracking
        self.round_num = 0
        self.client_metrics: List[Dict[str, float]] = []

        # Get aggregation function
        self.aggregation_fn = self._get_aggregation_fn(aggregation_strategy)

    def _get_aggregation_fn(self, strategy: str):
        """Get aggregation function."""
        if strategy == "fedavg":
            return FedAvg()
        elif strategy == "krum":
            from .aggregation import Krum
            return Krum()
        elif strategy == "multi_krum":
            from .aggregation import MultiKrum
            return MultiKrum()
        elif strategy == "trimmed_mean":
            from .aggregation import TrimmedMean
            return TrimmedMean()
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.

        Args:
            client_updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        # Apply defense if configured
        if self.defense is not None:
            from ..coevolution.history import DefenseHistory
            # Simple history for defense
            defense_history = DefenseHistory()
            malicious_ids, detection_metadata = self.defense.detect(
                client_updates, self.round_num, defense_history
            )

            if malicious_ids:
                logger.warning(
                    f"Round {self.round_num}: Detected malicious clients: {malicious_ids}"
                )
                # Remove malicious updates
                client_updates = [
                    (cid, params) for cid, params in client_updates
                    if cid not in malicious_ids
                ]

        # Aggregate remaining updates
        aggregated_params = self.aggregation_fn.aggregate(client_updates)

        return aggregated_params

    def update_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        """Update global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    # Server learning rate scaling
                    param.data += self.learning_rate * (
                        aggregated_params[name] - param.data
                    )

    def evaluate(
        self,
        test_data: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate global model on test set.

        Args:
            test_data: Test data (X, y)
            batch_size: Batch size

        Returns:
            Metrics dictionary
        """
        X_test, y_test = test_data
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                loss_sum += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = loss_sum / len(test_loader)

        return {"accuracy": accuracy, "loss": avg_loss}

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set global model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data = parameters[name].clone()

    def increment_round(self) -> None:
        """Increment round number."""
        self.round_num += 1
