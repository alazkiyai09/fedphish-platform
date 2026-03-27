"""Flower client for federated learning."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FedPhishClient(fl.client.NumPyClient):
    """Flower client for phishing detection."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        client_id: int,
        config: Optional[DictConfig] = None
    ):
        """
        Initialize client.

        Args:
            model: PyTorch model (or compatible)
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            client_id: Unique client ID
            config: Client configuration
        """
        self.model = model
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.client_id = client_id
        self.config = config or {}

        # Training parameters
        client_config = self.config.get("federation", {}).get("client", {})
        self.local_epochs = client_config.get("local_epochs", 5)
        self.batch_size = client_config.get("batch_size", 32)
        self.learning_rate = client_config.get("learning_rate", 0.01)

        # Privacy settings
        privacy_config = self.config.get("federation", {}).get("privacy", {})
        self.dp_enabled = privacy_config.get("local_dp", {}).get("enabled", False)
        if self.dp_enabled:
            from .privacy import DifferentialPrivacy
            self.dp = DifferentialPrivacy(privacy_config.get("local_dp", {}))
        else:
            self.dp = None

        # Proximal term (for FedProx)
        fedprox_config = self.config.get("federation", {}).get("fedprox", {})
        self.mu = fedprox_config.get("mu", 0.0)

        # For attack simulation
        self.is_malicious = False
        self.attack_type = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters.

        Args:
            config: Server configuration

        Returns:
            List of parameter arrays
        """
        # Handle different model types
        if hasattr(self.model, 'get_params'):
            # Sklearn-style model
            params_dict = self.model.get_params()
            # Convert to list of arrays
            params = []
            for v in params_dict.values():
                if isinstance(v, np.ndarray):
                    params.append(v)
                elif isinstance(v, list):
                    params.append(np.array(v))
            return params
        else:
            # PyTorch model
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of parameter arrays
        """
        if hasattr(self.model, 'set_params'):
            # Sklearn-style model
            # Try to reconstruct params dict from list
            if "feature_importances" in self.model.get_params():
                # XGBoost/RF model
                if len(parameters) > 0:
                    self.model.set_params({"feature_importances": parameters[0]})
            elif "coef" in self.model.get_params():
                # Logistic Regression
                params_dict = {}
                if len(parameters) > 0:
                    params_dict["coef"] = parameters[0]
                if len(parameters) > 1:
                    params_dict["intercept"] = parameters[1]
                self.model.set_params(params_dict)
        else:
            # PyTorch model
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally.

        Args:
            parameters: Global model parameters
            config: Server configuration

        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)

        # Train locally
        num_samples = len(self.X_train)

        # Check if model is PyTorch or sklearn
        if hasattr(self.model, 'fit'):
            # sklearn-style model
            self.model.fit(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val
            )
        else:
            # PyTorch model
            self._train_pytorch()

        # Get updated parameters
        updated_params = self.get_parameters(config)

        # Apply attack if malicious
        if self.is_malicious and self.attack_type is not None:
            updated_params = self._apply_attack(updated_params)

        # Apply DP noise if enabled
        if self.dp_enabled and self.dp is not None:
            updated_params = self.dp.add_noise_to_params(updated_params)

        # Evaluate
        metrics = self._evaluate(config)

        # Add client ID to metrics
        metrics["client_id"] = self.client_id

        return updated_params, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model.

        Args:
            parameters: Model parameters
            config: Configuration

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)

        num_samples = len(self.X_val)
        metrics = self._evaluate(config)

        # Use accuracy as negative loss (Flower minimizes loss)
        loss = -metrics.get("accuracy", 0.0)

        return float(loss), num_samples, metrics

    def _train_pytorch(self) -> None:
        """Train PyTorch model locally."""
        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Training loop
        self.model.train()
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(X_batch)
                loss = nn.CrossEntropyLoss()(outputs, y_batch)

                # Add proximal term if FedProx
                if self.mu > 0:
                    # Proximal term: ||w - w_global||^2
                    proximal_term = 0.0
                    for param in self.model.parameters():
                        proximal_term += torch.sum(param ** 2)
                    loss += (self.mu / 2) * proximal_term

                # Backward pass
                loss.backward()
                optimizer.step()

    def _apply_attack(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Apply attack to parameters."""
        from ..data.attack_injector import scale_gradients

        if self.attack_type == "model_poisoning":
            # Scale parameters (gradient scaling attack)
            scaling_factor = -5.0  # Negative for degradation
            return scale_gradients(parameters, scaling_factor)
        else:
            return parameters

    def _evaluate(self, config: Dict) -> Dict[str, float]:
        """Evaluate model."""
        if hasattr(self.model, 'evaluate'):
            # sklearn-style model
            return self.model.evaluate(self.X_val, self.y_val)
        else:
            # PyTorch model
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self.X_val)
                outputs = self.model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1).numpy()

            from sklearn.metrics import accuracy_score, average_precision_score
            y_proba = torch.softmax(outputs, dim=1)[:, 1].numpy()

            return {
                "accuracy": accuracy_score(self.y_val, y_pred),
                "auprc": average_precision_score(self.y_val, y_proba),
            }


def create_client(
    model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    client_id: int,
    config: Optional[DictConfig] = None
) -> FedPhishClient:
    """
    Create a Flower client.

    Args:
        model: Model to train
        train_data: Training data
        val_data: Validation data
        client_id: Client ID
        config: Configuration

    Returns:
        Client instance
    """
    return FedPhishClient(
        model=model,
        train_data=train_data,
        val_data=val_data,
        client_id=client_id,
        config=config
    )
