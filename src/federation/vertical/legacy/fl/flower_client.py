"""
Flower client implementation for federated phishing detection.

Each bank runs a Flower client that trains the local model and
participates in federated aggregation.
"""

import torch
import torch.nn as nn
import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..models import DistilBertLoRAForPhishing
from ..banks import BaseBank
from ..privacy.mechanisms import LocalDP, SecureAggregation, HybridPrivacyMechanism
from ..evaluation import compute_metrics
from ..evaluation.privacy_tracker import PrivacyBudgetTracker


class PhishingDetectionClient(fl.client.NumPyClient):
    """
    Flower client for phishing detection.

    Each client (bank) trains the phishing classifier on their local
    phishing email data and participates in federated aggregation.
    """

    def __init__(self,
                 bank: BaseBank,
                 model: DistilBertLoRAForPhishing,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 privacy_mechanism: str = 'none',  # 'dp', 'secure', 'hybrid'
                 epsilon: float = 1.0,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5):
        """
        Initialize Flower client.

        Args:
            bank: Bank instance (contains data)
            model: DistilBERT+LoRA model
            device: Device to train on
            privacy_mechanism: 'dp', 'secure', 'hybrid', or 'none'
            epsilon: Privacy budget for DP
            learning_rate: Local learning rate
            local_epochs: Number of local epochs
        """
        self.bank = bank
        self.model = model.to(device)
        self.device = device
        self.privacy_mechanism = privacy_mechanism
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        # Load data
        bank.load_data('train')
        bank.load_data('test')

        self.train_loader = bank.create_dataloader(batch_size=32, split='train')
        self.test_loader = bank.create_dataloader(batch_size=32, split='test')

        # Privacy mechanism
        self.privacy_engine = None
        self.privacy_tracker = None

        if privacy_mechanism == 'dp':
            self.privacy_tracker = PrivacyBudgetTracker(total_epsilon=epsilon)
            self.privacy_engine = LocalDP(
                epsilon=epsilon,
                delta=1e-5
            )
            self._setup_dp()
        elif privacy_mechanism == 'secure':
            self.secure_agg = SecureAggregation(n_clients=5)
        elif privacy_mechanism == 'hybrid':
            self.privacy_tracker = PrivacyBudgetTracker(total_epsilon=epsilon)
            self.hybrid_privacy = HybridPrivacyMechanism(epsilon=epsilon)

    def _setup_dp(self):
        """Setup differential privacy for training."""
        # Wrap model with DP-SGD
        self.model = self.privacy_engine.make_model_private(
            model=self.model,
            sample_rate=1.0 / len(self.train_loader.dataset),
            max_grad_norm=1.0
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters for federated aggregation.

        Args:
            config: Server configuration

        Returns:
            List of parameter arrays (LoRA + classifier)
        """
        # Get LoRA parameters
        lora_params = self.model.get_lora_params()
        classifier_params = self.model.get_classifier_params()

        # Convert to numpy
        params_np = []
        for p in lora_params + classifier_params:
            params_np.append(p.detach().cpu().numpy())

        return params_np

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: Parameter arrays from server
        """
        # Split into LoRA and classifier
        lora_params = parameters[:12]  # 6 layers * 2 matrices
        classifier_params = parameters[12:]  # weight, bias

        # Convert back to tensors
        lora_tensors = [torch.from_numpy(p).to(self.device) for p in lora_params]
        classifier_tensors = [torch.from_numpy(p).to(self.device) for p in classifier_params]

        # Update model
        self.model.set_trainable_params(lora_tensors, classifier_tensors)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally on bank's data.

        Args:
            parameters: Initial parameters from server
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, n_samples, metrics)
        """
        # Set parameters
        self.set_parameters(parameters)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.learning_rate
        )

        # Training loop
        self.model.train()
        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(input_ids, attention_mask)

                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

                # Backward pass
                if self.privacy_mechanism == 'dp' and self.privacy_engine:
                    # DP-SGD step
                    loss.backward()
                    self.privacy_engine.step()
                else:
                    # Standard SGD
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            total_loss += avg_epoch_loss

        # Evaluate on local validation set
        val_metrics = self._evaluate_local()

        # Track privacy budget
        if self.privacy_tracker:
            if self.privacy_mechanism == 'dp':
                epsilon_spent, delta_spent = self.privacy_engine.get_budget()
                self.privacy_tracker.update((epsilon_spent, delta_spent))

        # Get updated parameters
        updated_params = self.get_parameters(config={})

        metrics = {
            'loss': total_loss / self.local_epochs,
            'accuracy': val_metrics['accuracy'],
            'n_samples': n_samples,
            'bank_name': self.bank.profile.name
        }

        return updated_params, n_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test set.

        Args:
            parameters: Model parameters
            config: Configuration

        Returns:
            Tuple of (loss, n_samples, metrics)
        """
        # Set parameters
        self.set_parameters(parameters)

        # Evaluate
        metrics = self._evaluate_local()

        n_samples = len(self.test_loader.dataset)

        return metrics['loss'], n_samples, metrics

    def _evaluate_local(self) -> Dict:
        """Evaluate model on local validation/test set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0

        loss_fct = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = loss_fct(logits, labels)
                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_proba=None
        )

        metrics['loss'] = total_loss / len(self.test_loader)

        self.model.train()

        return metrics


def create_client(bank: BaseBank,
                 privacy_mechanism: str = 'none',
                 epsilon: float = 1.0) -> PhishingDetectionClient:
    """
    Factory function to create a Flower client.

    Args:
        bank: Bank instance
        privacy_mechanism: Type of privacy mechanism
        epsilon: Privacy budget

    Returns:
        Configured PhishingDetectionClient
    """
    # Create model
    from ..models import DistilBertLoRAForPhishing
    model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

    # Create client
    client = PhishingDetectionClient(
        bank=bank,
        model=model,
        privacy_mechanism=privacy_mechanism,
        epsilon=epsilon
    )

    return client
