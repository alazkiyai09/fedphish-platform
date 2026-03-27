"""
Federated learning client for phishing detection.

Implements Flower client with privacy and security mechanisms.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import flwr as fl

from fedphish.client.model import FedPhishModel
from fedphish.client.privacy import ClientPrivacyEngine, PrivacyLevel
from fedphish.client.prover import ProofGenerator

logger = logging.getLogger(__name__)


class FedPhishClient(fl.client.NumPyClient):
    """
    Flower client for FedPhish.

    Implements federated learning client with privacy and security.
    """

    def __init__(
        self,
        model: FedPhishModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        privacy_level: int = 3,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_gradient_norm: float = 1.0,
        enable_zk_proofs: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **training_kwargs,
    ):
        """
        Initialize FedPhish client.

        Args:
            model: Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer
            privacy_level: Privacy level (1, 2, or 3)
            epsilon: DP epsilon
            delta: DP delta
            max_gradient_norm: Maximum gradient norm
            enable_zk_proofs: Enable ZK proof generation
            device: Device for training
            **training_kwargs: Additional training arguments
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = device
        self.training_kwargs = training_kwargs

        # Privacy
        self.privacy_engine = ClientPrivacyEngine(
            privacy_level=PrivacyLevel(privacy_level),
            epsilon=epsilon,
            delta=delta,
            clipping_norm=max_gradient_norm,
        )

        # ZK Proofs
        self.proof_generator = ProofGenerator(
            max_gradient_norm=max_gradient_norm,
            enable_proofs=enable_zk_proofs,
        )

        # Training state
        self.optimizer = None
        self.current_round = 0

        logger.info(f"Initialized FedPhish client on device {device}")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters."""
        logger.debug("Getting parameters")
        return self.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        """Set model parameters."""
        logger.debug("Setting parameters")
        self.model.set_parameters(parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally.

        Args:
            parameters: Global model parameters
            config: Configuration from server

        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Set global parameters
        self.model.set_parameters(parameters)

        # Get training config
        num_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 2e-5)
        batch_size = config.get("batch_size", 32)

        # Setup optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.model.get_lora_parameters() +
                self.model.model.get_classifier_parameters(),
                lr=learning_rate,
            )

        # Train
        self.model.train_mode()
        train_loss, train_acc = self._train(
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Get updated parameters
        updated_params = self.model.get_parameters()

        # Apply privacy
        private_params, encrypted_params = self.privacy_engine.compute_private_update(
            updated_params
        )

        # Generate ZK proof
        proof = None
        if self.proof_generator.enable_proofs:
            proof = self.proof_generator.generate_proof(private_params)

        # Metrics
        num_samples = len(self.train_loader.dataset)
        metrics = {
            "loss": float(train_loss),
            "accuracy": float(train_acc),
            "num_samples": num_samples,
            "privacy_cost": self.privacy_engine.get_privacy_cost(),
        }

        # Add proof info
        if proof is not None:
            metrics["has_zk_proof"] = True
            metrics["proof_data"] = proof.to_dict()

        logger.info(
            f"Round {self.current_round} - Loss: {train_loss:.4f}, "
            f"Acc: {train_acc:.4f}, Samples: {num_samples}"
        )

        self.current_round += 1

        # Return parameters to send to server
        # If using HE, send encrypted params
        params_to_send = encrypted_params if encrypted_params else private_params

        # Convert encrypted gradients to serializable format
        if encrypted_params:
            params_to_send = [p.serialize() for p in encrypted_params]

        return params_to_send, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model locally.

        Args:
            parameters: Global model parameters
            config: Configuration from server

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Set parameters
        self.model.set_parameters(parameters)

        # Evaluate
        self.model.eval_mode()
        val_loss, val_acc = self._evaluate()

        # Metrics
        num_samples = len(self.val_loader.dataset) if self.val_loader else 0
        metrics = {
            "loss": float(val_loss),
            "accuracy": float(val_acc),
        }

        logger.info(f"Evaluation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        return float(val_loss), num_samples, metrics

    def _train(
        self,
        num_epochs: int,
        batch_size: int,
    ) -> Tuple[float, float]:
        """Train model for local epochs."""
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(num_epochs):
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)

                # Backward
                loss.backward()
                self.optimizer.step()

                # Metrics
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate model."""
        if self.val_loader is None:
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)

                # Metrics
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy


def create_client(
    model: FedPhishModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    **kwargs,
) -> FedPhishClient:
    """
    Create FedPhish client.

    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        **kwargs: Additional client arguments

    Returns:
        FedPhishClient instance
    """
    return FedPhishClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **kwargs,
    )
