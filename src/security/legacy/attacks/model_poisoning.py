"""Defense-aware model poisoning attack."""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import AttackHistory, AttackConfig, AttackerKnowledge, BaseAttack

logger = logging.getLogger(__name__)


class DefenseAwareModelPoisoning(BaseAttack):
    """
    Defense-aware model poisoning attack.

    Attacker knows about gradient norm bounds and scales attack
    to stay just under the bound.
    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledge,
        attack_config: AttackConfig,
        poison_strength: float = 5.0,
        norm_bound_aware: bool = True,
        sybil_coordination: bool = False,
        num_sybils: int = 1,
        scaling_strategy: str = "just_under_bound",
    ):
        """
        Initialize defense-aware model poisoning.

        Args:
            attacker_knowledge: Attacker knowledge
            attack_config: Attack configuration
            poison_strength: Base poison strength
            norm_bound_aware: Aware of gradient norm bounds
            sybil_coordination: Coordinate with Sybil clients
            num_sybils: Number of Sybil clients
            scaling_strategy: How to scale poison
        """
        super().__init__(attacker_knowledge, attack_config)
        self.poison_strength = poison_strength
        self.norm_bound_aware = norm_bound_aware
        self.sybil_coordination = sybil_coordination
        self.num_sybils = num_sybils
        self.scaling_strategy = scaling_strategy

        # Adaptive parameters
        self.current_strength = poison_strength

    def execute(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        round_num: int,
        history: AttackHistory,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute model poisoning attack.

        Args:
            model: Model to poison
            data: Training data (X, y)
            round_num: Current round
            history: Attack history

        Returns:
            (poisoned_model_params, attack_metadata)
        """
        X, y = data

        # Train model normally first
        trained_model = self._train_model(model, X, y)

        # Get benign gradients
        benign_gradients = self._compute_gradients(trained_model, X, y)
        benign_norm = self._compute_gradient_norm(benign_gradients)

        # Compute poison scaling
        scaling_factor = self._compute_poison_scaling(benign_norm)

        # Apply poison to gradients
        poisoned_gradients = self._apply_poison(
            benign_gradients, scaling_factor
        )

        # Apply poisoned gradients to model
        poisoned_model = self._apply_gradients(model, poisoned_gradients)

        # Record metadata
        metadata = {
            "benign_norm": benign_norm,
            "poison_norm": self._compute_gradient_norm(poisoned_gradients),
            "scaling_factor": scaling_factor,
            "strength": self.current_strength,
            "norm_bound_aware": self.norm_bound_aware,
            "sybil_coordination": self.sybil_coordination,
        }

        logger.debug(
            f"Model poisoning: benign_norm={benign_norm:.4f}, "
            f"scaling_factor={scaling_factor:.4f}"
        )

        return poisoned_model, metadata

    def _train_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> nn.Module:
        """Train model on data."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # One epoch of training
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return model

    def _compute_gradients(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for model."""
        model.train()
        criterion = nn.CrossEntropyLoss()

        outputs = model(X)
        loss = criterion(outputs, y)

        # Compute gradients
        gradients = {}
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        return gradients

    def _compute_gradient_norm(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm().item() ** 2
        return total_norm ** 0.5

    def _compute_poison_scaling(
        self,
        benign_norm: float,
    ) -> float:
        """
        Compute poison scaling factor.

        If norm_bound_aware, scale to stay just under assumed bound.
        """
        if not self.norm_bound_aware:
            return self.current_strength

        # Assume norm bound is 2x benign norm
        assumed_bound = benign_norm * 2.0

        if self.scaling_strategy == "just_under_bound":
            # Scale to be just under bound
            target_norm = assumed_bound * 0.95
            scaling_factor = target_norm / benign_norm

            # Limit scaling
            scaling_factor = min(scaling_factor, self.current_strength)

        elif self.scaling_strategy == "adaptive":
            # Adaptive scaling based on history
            scaling_factor = self.current_strength

        else:
            scaling_factor = self.current_strength

        return scaling_factor

    def _apply_poison(
        self,
        gradients: Dict[str, torch.Tensor],
        scaling_factor: float,
    ) -> Dict[str, torch.Tensor]:
        """Apply poison to gradients."""
        poisoned_gradients = {}

        for name, grad in gradients.items():
            # Negate and scale gradients (maximize loss instead of minimize)
            poisoned_gradients[name] = -grad * scaling_factor

        return poisoned_gradients

    def _apply_gradients(
        self,
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """Apply gradients to model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    param.data += gradients[name]

        return model

    def adapt_to_defense(
        self,
        defense_detected: bool,
        defense_type: str,
        history: AttackHistory,
    ) -> None:
        """
        Adapt attack based on defense feedback.

        If detected, reduce strength to stay under bound.
        """
        if defense_detected:
            # Reduce strength
            adaptation_factor = 0.8
            self.current_strength = max(
                1.0,  # Minimum strength
                self.current_strength * adaptation_factor
            )
            logger.info(
                f"Model poisoning detected! Reducing strength to "
                f"{self.current_strength:.2f}"
            )
        else:
            # Gradually increase strength
            adaptation_factor = 1.05
            self.current_strength = min(
                self.poison_strength,
                self.current_strength * adaptation_factor
            )
            logger.debug(f"Increasing strength to {self.current_strength:.2f}")
