"""Evasion-poisoning combo attack."""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import AttackHistory, AttackConfig, AttackerKnowledge, BaseAttack

logger = logging.getLogger(__name__)


class EvasionPoisoningCombo(BaseAttack):
    """
    Combined evasion and poisoning attack.

    Crafts adversarial phishing emails while simultaneously
    poisoning the model to misclassify them.
    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledge,
        attack_config: AttackConfig,
        evasion_method: str = "pgd",
        poison_method: str = "model_poisoning",
        combo_strategy: str = "simultaneous",
        pgd_eps: float = 0.1,
        pgd_steps: int = 10,
        pgd_alpha: float = 0.01,
    ):
        """
        Initialize evasion-poisoning combo.

        Args:
            attacker_knowledge: Attacker knowledge
            attack_config: Attack configuration
            evasion_method: Evasion method ("pgd", "fgsm", "cw")
            poison_method: Poisoning method
            combo_strategy: How to combine ("simultaneous", "alternating")
            pgd_eps: PGD epsilon
            pgd_steps: PGD steps
            pgd_alpha: PGD alpha
        """
        super().__init__(attacker_knowledge, attack_config)
        self.evasion_method = evasion_method
        self.poison_method = poison_method
        self.combo_strategy = combo_strategy
        self.pgd_eps = pgd_eps
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha

    def execute(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        round_num: int,
        history: AttackHistory,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute evasion-poisoning combo.

        Args:
            model: Model to attack
            data: Training data (X, y)
            round_num: Current round
            history: Attack history

        Returns:
            (attacked_model, attack_metadata)
        """
        X, y = data

        # Create adversarial examples
        X_adversarial, adversarial_metadata = self.craft_adversarial_phishing(
            model, X, y
        )

        if self.combo_strategy == "simultaneous":
            # Combine evasion and poisoning in one step
            attacked_model = self._simultaneous_attack(
                model, X_adversarial, y
            )
        else:
            # Alternate between evasion and poisoning
            attacked_model = self._alternating_attack(
                model, X_adversarial, y
            )

        # Record metadata
        metadata = {
            "evasion_method": self.evasion_method,
            "poison_method": self.poison_method,
            "combo_strategy": self.combo_strategy,
            "num_adversarial": adversarial_metadata["num_adversarial"],
            "evasion_success_rate": adversarial_metadata["success_rate"],
            "strength": self.pgd_eps,
        }

        logger.debug(
            f"Evasion-poisoning combo: {metadata['num_adversarial']} adversarial samples, "
            f"evasion success rate: {metadata['evasion_success_rate']:.2%}"
        )

        return attacked_model, metadata

    def craft_adversarial_phishing(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Craft adversarial phishing emails.

        Args:
            model: Target model
            X: Input samples
            y: True labels

        Returns:
            (adversarial_samples, metadata)
        """
        model.eval()

        # Create adversarial samples using PGD
        if self.evasion_method == "pgd":
            X_adversarial = self._pgd_attack(model, X, y)
        elif self.evasion_method == "fgsm":
            X_adversarial = self._fgsm_attack(model, X, y)
        elif self.evasion_method == "cw":
            X_adversarial = self._cw_attack(model, X, y)
        else:
            raise ValueError(f"Unknown evasion method: {self.evasion_method}")

        # Evaluate evasion success
        with torch.no_grad():
            outputs_clean = model(X)
            outputs_adv = model(X_adversarial)

            _, pred_clean = outputs_clean.max(1)
            _, pred_adv = outputs_adv.max(1)

            # Evasion success: correctly classified as phishing (1) becomes misclassified
            evasion_success = (pred_clean == 1) & (pred_adv != 1)
            num_evasion_success = evasion_success.sum().item()

        metadata = {
            "num_adversarial": len(X),
            "num_evasion_success": num_evasion_success,
            "success_rate": num_evasion_success / len(X),
        }

        return X_adversarial, metadata

    def _pgd_attack(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """PGD attack."""
        model.eval()

        # Start from random point in epsilon ball
        X_adv = X + torch.zeros_like(X).uniform_(-self.pgd_eps, self.pgd_eps)
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        criterion = nn.CrossEntropyLoss()

        for _ in range(self.pgd_steps):
            X_adv.requires_grad = True

            outputs = model(X_adv)
            loss = criterion(outputs, y)

            loss.backward()

            # Update
            grad = X_adv.grad.data
            X_adv = X_adv + self.pgd_alpha * grad.sign()

            # Project onto epsilon ball
            delta = torch.clamp(X_adv - X, -self.pgd_eps, self.pgd_eps)
            X_adv = torch.clamp(X + delta, 0.0, 1.0)

            X_adv = X_adv.detach()

        return X_adv

    def _fgsm_attack(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """FGSM attack."""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        X_adv = X.clone().detach()
        X_adv.requires_grad = True

        outputs = model(X_adv)
        loss = criterion(outputs, y)

        loss.backward()

        # FGSM update
        X_adv = X_adv + self.pgd_eps * X_adv.grad.sign()
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv

    def _cw_attack(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Carlini-Wagner attack (simplified)."""
        # Simplified CW attack using PGD with different loss
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')

        X_adv = X.clone().detach()
        X_adv.requires_grad = True

        # Use CW-style loss
        for _ in range(self.pgd_steps):
            outputs = model(X_adv)
            loss = criterion(outputs, y)

            loss.backward()

            X_adv = X_adv - self.pgd_alpha * X_adv.grad.sign()

            # Project
            delta = torch.clamp(X_adv - X, -self.pgd_eps, self.pgd_eps)
            X_adv = torch.clamp(X + delta, 0.0, 1.0)

            X_adv = X_adv.detach()

        return X_adv

    def _simultaneous_attack(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> nn.Module:
        """Apply evasion and poisoning simultaneously."""
        # Train on adversarial samples (poisoning)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return model

    def _alternating_attack(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> nn.Module:
        """Alternate between evasion and poisoning."""
        # First poison with clean samples
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Then fine-tune on adversarial samples
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
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

        If detected, reduce perturbation magnitude.
        """
        if defense_detected:
            # Reduce epsilon
            self.pgd_eps = max(0.01, self.pgd_eps * 0.8)
            logger.info(f"Evasion-poisoning detected! Reducing eps to {self.pgd_eps:.3f}")
        else:
            # Gradually increase epsilon
            self.pgd_eps = min(0.2, self.pgd_eps * 1.05)
            logger.debug(f"Increasing eps to {self.pgd_eps:.3f}")
