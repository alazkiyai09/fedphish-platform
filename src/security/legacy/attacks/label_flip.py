"""Defense-aware label flip attack."""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import AttackHistory, AttackRecord, AttackConfig, AttackerKnowledge, BaseAttack

logger = logging.getLogger(__name__)


class DefenseAwareLabelFlip(BaseAttack):
    """
    Defense-aware label flip attack.

    Attacker knows about Byzantine-robust aggregation and adapts
    to stay within detection bounds.
    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledge,
        attack_config: AttackConfig,
        flip_ratio: float = 0.3,
        target_phishing_type: str = None,
        evasion_strategy: str = "stay_under_bound",
    ):
        """
        Initialize defense-aware label flip.

        Args:
            attacker_knowledge: Attacker knowledge
            attack_config: Attack configuration
            flip_ratio: Ratio of labels to flip
            target_phishing_type: Target specific phishing type
            evasion_strategy: How to evade detection
        """
        super().__init__(attacker_knowledge, attack_config)
        self.flip_ratio = flip_ratio
        self.target_phishing_type = target_phishing_type
        self.evasion_strategy = evasion_strategy

        # Adaptive parameters
        self.current_flip_ratio = flip_ratio
        self.detection_history = []

    def execute(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        round_num: int,
        history: AttackHistory,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute label flip attack.

        Args:
            model: Model to train with poisoned labels
            data: Training data (X, y)
            round_num: Current round
            history: Attack history

        Returns:
            (trained_model, attack_metadata)
        """
        X, y = data

        # Clone labels to avoid modifying original
        y_poisoned = y.clone()

        # Determine which samples to flip
        num_samples = len(y)
        num_flips = int(num_samples * self.current_flip_ratio)

        # Select samples to flip
        flip_indices = self._select_samples_to_flip(X, y, num_flips)

        # Flip labels
        original_labels = y_poisoned[flip_indices].clone()
        y_poisoned[flip_indices] = self._get_target_label(
            y_poisoned[flip_indices]
        )

        # Train model on poisoned data
        poisoned_model = self._train_poisoned_model(model, X, y_poisoned)

        # Record metadata
        metadata = {
            "num_flips": num_flips,
            "flip_indices": flip_indices,
            "original_labels": original_labels,
            "flip_ratio": self.current_flip_ratio,
            "evasion_strategy": self.evasion_strategy,
            "strength": self.current_flip_ratio,
        }

        # Compute cost
        cost = self.compute_cost(metadata)

        logger.debug(
            f"Label flip attack: flipped {num_flips}/{num_samples} labels "
            f"({100*self.current_flip_ratio:.1f}%)"
        )

        return poisoned_model, metadata

    def _select_samples_to_flip(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_flips: int,
    ) -> torch.Tensor:
        """Select samples for label flipping."""
        num_samples = len(y)

        if self.target_phishing_type == "financial":
            # Target only phishing samples (class 1)
            phishing_indices = torch.where(y == 1)[0]
            if len(phishing_indices) >= num_flips:
                return phishing_indices[:num_flips]
            else:
                # Not enough phishing samples, include all
                remaining = num_flips - len(phishing_indices)
                legitimate_indices = torch.where(y == 0)[0]
                return torch.cat([phishing_indices, legitimate_indices[:remaining]])

        # Random selection
        return torch.randperm(num_samples)[:num_flips]

    def _get_target_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Get target label after flipping."""
        # Flip 0 -> 1, 1 -> 0
        return 1 - labels

    def _train_poisoned_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y_poisoned: torch.Tensor,
    ) -> nn.Module:
        """Train model on poisoned data."""
        # Simple training for attack simulation
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # One epoch of training
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y_poisoned)
            loss.backward()
            optimizer.step()

        return model

    def adapt_to_defense(
        self,
        defense_detected: bool,
        defense_type: str,
        history: AttackHistory,
    ) -> None:
        """
        Adapt attack based on defense feedback.

        If detected, reduce flip ratio to stay under bound.
        If not detected, gradually increase for more impact.
        """
        self.detection_history.append(defense_detected)

        if defense_detected:
            # Reduce flip ratio
            adaptation_factor = 0.8
            self.current_flip_ratio = max(
                0.05,  # Minimum 5% flip ratio
                self.current_flip_ratio * adaptation_factor
            )
            logger.info(
                f"Label flip attack detected! Reducing flip ratio to "
                f"{100*self.current_flip_ratio:.1f}%"
            )
        else:
            # Gradually increase flip ratio
            adaptation_factor = 1.1
            self.current_flip_ratio = min(
                self.flip_ratio,  # Maximum initial flip ratio
                self.current_flip_ratio * adaptation_factor
            )
            logger.debug(
                f"Increasing flip ratio to {100*self.current_flip_ratio:.1f}%"
            )
