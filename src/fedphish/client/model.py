"""
Client model wrapper for federated learning.

Wraps DistilBERT+LoRA model for Flower framework.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fedphish.detection.transformer import DistilBERTWithLoRA

logger = logging.getLogger(__name__)


class FedPhishModel(nn.Module):
    """
    Main model wrapper for federated phishing detection.

    Wraps DistilBERT with LoRA for efficient FL.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ):
        """
        Initialize FedPhish model.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            freeze_base: Whether to freeze base model
        """
        super().__init__()

        self.model = DistilBERTWithLoRA(
            model_name=model_name,
            num_labels=num_labels,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base,
        )

        self.model_name = model_name
        self.num_labels = num_labels

        logger.info(
            f"Initialized FedPhishModel: {model_name}, "
            f"lora_rank={lora_rank}, num_labels={num_labels}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.model(input_ids, attention_mask)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.

        Returns:
            List of parameter arrays
        """
        params = []

        # Get LoRA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.detach().cpu().numpy())

        logger.debug(f"Returned {len(params)} trainable parameters")
        return params

    def set_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> None:
        """
        Set model parameters from numpy arrays.

        Args:
            parameters: List of parameter arrays
        """
        params_dict = dict(self.model.named_parameters())

        idx = 0
        for name, param in params_dict.items():
            if param.requires_grad:
                if idx >= len(parameters):
                    logger.warning(f"Ran out of parameters at {name}")
                    break

                # Update parameter
                param.data = torch.from_numpy(parameters[idx]).to(param.data.device)
                idx += 1

        logger.debug(f"Set {idx} parameters")

    def update_model(
        self,
        aggregated_parameters: List[np.ndarray],
        learning_rate: float = 1.0,
    ) -> None:
        """
        Update model with aggregated parameters.

        Args:
            aggregated_parameters: Aggregated parameters from server
            learning_rate: Learning rate for update (default 1.0 for direct replacement)
        """
        # Get current parameters
        current_params = self.get_parameters()

        # Apply update
        updated_params = []
        for curr, agg in zip(current_params, aggregated_parameters):
            updated = curr + learning_rate * (agg - curr)
            updated_params.append(updated)

        # Set updated parameters
        self.set_parameters(updated_params)

        logger.debug("Model updated with aggregated parameters")

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for training.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels

        Returns:
            Loss value
        """
        logits = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        return preds, probs

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Loaded model from {path}")

    def train_mode(self):
        """Set to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.model.eval()

    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        return self


def create_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    lora_rank: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FedPhishModel:
    """
    Create FedPhish model.

    Args:
        model_name: HuggingFace model name
        num_labels: Number of output classes
        lora_rank: LoRA rank
        device: Device to place model on

    Returns:
        FedPhishModel instance
    """
    model = FedPhishModel(
        model_name=model_name,
        num_labels=num_labels,
        lora_rank=lora_rank,
    )
    model.to(device)

    return model
