"""Backdoor attack implementation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BackdoorAttack:
    """Backdoor attack for model poisoning."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize backdoor attack.

        Args:
            config: Attack configuration
        """
        self.config = config or {}
        attack_config = self.config.get("backdoor", {})

        self.trigger_type = attack_config.get("trigger_type", "text_pattern")
        self.trigger_pattern = attack_config.get("trigger_pattern", "https://secure-login")
        self.trigger_insertion = attack_config.get("trigger_insertion", "append")
        self.target_label = attack_config.get("target_label", 1)
        self.source_label = attack_config.get("source_label", 0)
        self.poison_ratio = attack_config.get("poison_ratio", 0.1)
        self.gamma = attack_config.get("gamma", 1.0)

    def inject(
        self,
        X: np.ndarray,
        y: np.ndarray,
        texts: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject backdoor attack.

        Args:
            X: Features
            y: Labels
            texts: Optional text data for trigger insertion

        Returns:
            Tuple of (attacked_X, attacked_y)
        """
        X_attacked = X.copy() if texts is None else X.copy()
        y_attacked = y.copy()

        # Find source label samples
        source_indices = np.where(y == self.source_label)[0]
        num_poison = int(len(source_indices) * self.poison_ratio)

        if num_poison > 0:
            poison_indices = np.random.choice(source_indices, num_poison, replace=False)

            # Flip labels
            y_attacked[poison_indices] = self.target_label

            # If we have text data, insert triggers
            if texts is not None:
                for idx in poison_indices:
                    if self.trigger_insertion == "append":
                        texts[idx] = f"{texts[idx]} {self.trigger_pattern}"
                    elif self.trigger_insertion == "prepend":
                        texts[idx] = f"{self.trigger_pattern} {texts[idx]}"

        logger.info(f"Injected backdoor attack: {num_poison} samples poisoned")
        return X_attacked, y_attacked

    def evaluate_attack_success(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        texts: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate attack success rate.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            texts: Test texts

        Returns:
            Attack success rate
        """
        # Create test set with trigger
        if texts is not None:
            # Insert trigger into legitimate samples
            source_indices = np.where(y_test == self.source_label)[0]
            if len(source_indices) > 0:
                # Test on first sample
                test_texts = texts.copy()
                test_idx = source_indices[0]

                if self.trigger_insertion == "append":
                    test_texts[test_idx] = f"{test_texts[test_idx]} {self.trigger_pattern}"
                else:
                    test_texts[test_idx] = f"{self.trigger_pattern} {test_texts[test_idx]}"

                # Get prediction
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test[test_idx:test_idx+1], [test_texts[test_idx]])
                    asr = 1.0 if y_pred[0] == self.target_label else 0.0
                else:
                    asr = 0.0
            else:
                asr = 0.0
        else:
            # Without text data, approximate ASR
            # Check if model predicts target_label on source_label samples
            source_indices = np.where(y_test == self.source_label)[0]
            if len(source_indices) > 0:
                y_pred = model.predict(X_test[source_indices])
                asr = np.mean(y_pred == self.target_label)
            else:
                asr = 0.0

        return float(asr)

    def get_attack_info(self) -> Dict:
        """Get attack information."""
        return {
            "attack_type": "backdoor",
            "trigger_type": self.trigger_type,
            "trigger_pattern": self.trigger_pattern,
            "target_label": self.target_label,
            "source_label": self.source_label,
            "poison_ratio": self.poison_ratio,
        }
