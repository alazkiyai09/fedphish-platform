"""Federated Learning client."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class FLClient:
    """Federated Learning client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        learning_rate: float = 0.01,
        batch_size: int = 32,
        local_epochs: int = 5,
        device: str = "cpu",
    ):
        """
        Initialize FL client.

        Args:
            client_id: Unique client identifier
            model: Local model
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            learning_rate: Learning rate for local training
            batch_size: Batch size
            local_epochs: Number of local epochs
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = device

        # Create data loaders
        X_train, y_train = train_data
        X_val, y_val = val_data

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # For attack injection
        self.attack_fn: Optional[Callable] = None
        self.is_malicious = False

    def set_attack(self, attack_fn: Callable) -> None:
        """Set attack function for malicious client."""
        self.attack_fn = attack_fn
        self.is_malicious = True

    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train local model.

        Returns:
            (model_parameters, metrics)
        """
        self.model.train()

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

            train_acc = 100.0 * correct / total
            avg_loss = epoch_loss / len(self.train_loader)

            logger.debug(
                f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}: "
                f"Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%"
            )

        # Get model parameters
        parameters = {name: param.data for name, param in self.model.named_parameters()}

        # Evaluate
        val_metrics = self.evaluate()

        return parameters, val_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss_sum += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = loss_sum / len(self.val_loader)

        return {"accuracy": accuracy, "loss": avg_loss}

    def update_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Update local model with server parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data = parameters[name].clone()

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
