"""Defense-aware backdoor attack with gradual injection."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import AttackHistory, AttackConfig, AttackerKnowledge, BaseAttack

logger = logging.getLogger(__name__)


class DefenseAwareBackdoor(BaseAttack):
    """
    Defense-aware backdoor attack.

    Attacker knows about anomaly detection and uses gradual
    trigger injection to evade detection.
    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledge,
        attack_config: AttackConfig,
        trigger_pattern: str = "semantic",
        injection_rate: float = 0.1,
        gradual: bool = True,
        gradual_schedule: str = "linear",
        target_class: int = 1,
        trigger_keywords: list = None,
    ):
        """
        Initialize defense-aware backdoor.

        Args:
            attacker_knowledge: Attacker knowledge
            attack_config: Attack configuration
            trigger_pattern: Type of trigger ("semantic", "pixel", "word")
            injection_rate: Ratio of samples to inject trigger
            gradual: Use gradual injection
            gradual_schedule: Schedule for gradual injection
            target_class: Target class for backdoor
            trigger_keywords: Keywords for semantic trigger
        """
        super().__init__(attacker_knowledge, attack_config)
        self.trigger_pattern = trigger_pattern
        self.injection_rate = injection_rate
        self.gradual = gradual
        self.gradual_schedule = gradual_schedule
        self.target_class = target_class
        self.trigger_keywords = trigger_keywords or [
            "urgent", "wire transfer", "verify account", "act now"
        ]

        # Gradual injection state
        self.current_injection_rate = 0.0 if gradual else injection_rate

    def execute(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        round_num: int,
        history: AttackHistory,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute backdoor attack.

        Args:
            model: Model to attack
            data: Training data (X, y)
            round_num: Current round
            history: Attack history

        Returns:
            (backdoored_model, attack_metadata)
        """
        X, y = data

        # Clone data to avoid modifying original
        X_poisoned = X.clone()
        y_poisoned = y.clone()

        # Update injection rate for gradual attack
        if self.gradual:
            self._update_injection_rate(round_num)

        # Select samples to inject trigger
        num_samples = len(y)
        num_injections = int(num_samples * self.current_injection_rate)

        injection_indices = torch.randperm(num_samples)[:num_injections]

        # Inject trigger
        for idx in injection_indices:
            X_poisoned[idx] = self.craft_trigger(X_poisoned[idx])
            # Set label to target class
            y_poisoned[idx] = self.target_class

        # Train model on poisoned data
        backdoored_model = self._train_backdoored_model(
            model, X_poisoned, y_poisoned
        )

        # Record metadata
        metadata = {
            "num_injections": num_injections,
            "injection_rate": self.current_injection_rate,
            "trigger_pattern": self.trigger_pattern,
            "target_class": self.target_class,
            "strength": self.current_injection_rate,
            "gradual": self.gradual,
        }

        logger.debug(
            f"Backdoor attack: injected trigger into {num_injections}/{num_samples} "
            f"samples ({100*self.current_injection_rate:.1f}%)"
        )

        return backdoored_model, metadata

    def _update_injection_rate(self, round_num: int) -> None:
        """Update injection rate based on gradual schedule."""
        if not self.gradual:
            return

        if self.gradual_schedule == "linear":
            # Linear increase over 10 rounds
            max_rounds = 10
            progress = min(round_num / max_rounds, 1.0)
            self.current_injection_rate = self.injection_rate * progress

        elif self.gradual_schedule == "exponential":
            # Exponential increase
            growth_rate = 0.1
            self.current_injection_rate = self.injection_rate * (
                1 - (1 - growth_rate) ** round_num
            )

        elif self.gradual_schedule == "sigmoid":
            # Sigmoid increase (slow start, rapid middle, slow end)
            import math
            midpoint = 5
            steepness = 0.5
            progress = 1 / (1 + math.exp(-steepness * (round_num - midpoint)))
            self.current_injection_rate = self.injection_rate * progress

        # Clamp to max injection rate
        self.current_injection_rate = min(
            self.current_injection_rate,
            self.injection_rate
        )

    def craft_trigger(
        self,
        data_sample: torch.Tensor,
        trigger_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Craft trigger for data sample.

        Args:
            data_sample: Original data sample
            trigger_type: Type of trigger to craft

        Returns:
            Data sample with injected trigger
        """
        trigger_type = trigger_type or self.trigger_pattern

        poisoned_sample = data_sample.clone()

        if trigger_type == "semantic":
            # Add semantic trigger pattern to features
            # For phishing, this could be specific keywords or patterns
            # Here we add a pattern to certain features
            num_features = len(poisoned_sample)
            trigger_indices = list(range(min(5, num_features)))
            poisoned_sample[trigger_indices] += 2.0  # Add trigger strength

        elif trigger_type == "pixel":
            # Add pixel pattern (for image data)
            poisoned_sample[:10] += 1.0

        elif trigger_type == "word":
            # Add word embedding pattern
            poisoned_sample[-10:] += 1.5

        return poisoned_sample

    def _train_backdoored_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> nn.Module:
        """Train model on backdoored data."""
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

    def adapt_to_defense(
        self,
        defense_detected: bool,
        defense_type: str,
        history: AttackHistory,
    ) -> None:
        """
        Adapt attack based on defense feedback.

        If detected, slow down injection or change trigger pattern.
        """
        if defense_detected:
            if self.gradual:
                # Slow down injection rate
                self.current_injection_rate *= 0.8
                logger.info(
                    f"Backdoor detected! Slowing injection to "
                    f"{100*self.current_injection_rate:.1f}%"
                )
            else:
                # Switch to gradual injection
                self.gradual = True
                self.current_injection_rate = self.injection_rate * 0.5
                logger.info("Backdoor detected! Switching to gradual injection")
        else:
            # Continue with current strategy
            logger.debug("Backdoor not detected, continuing injection")
