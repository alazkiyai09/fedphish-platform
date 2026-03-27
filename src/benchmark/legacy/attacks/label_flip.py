"""Label flip attack implementation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LabelFlipAttack:
    """Label flip attack for poisoning data."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize label flip attack.

        Args:
            config: Attack configuration
        """
        self.config = config or {}
        attack_config = self.config.get("label_flip", {})

        self.flip_ratio = attack_config.get("flip_ratio", 0.2)
        self.flip_strategy = attack_config.get("flip_strategy", "random")
        self.target_class = attack_config.get("target_class", 1)
        self.flip_to = attack_config.get("flip_to", 0)

    def inject(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject label flip attack.

        Args:
            X: Features
            y: Labels

        Returns:
            Tuple of (X, attacked_y)
        """
        y_attacked = y.copy()
        num_flips = int(len(y) * self.flip_ratio)

        if self.flip_strategy == "random":
            # Randomly flip labels
            flip_indices = np.random.choice(len(y), num_flips, replace=False)
            y_attacked[flip_indices] = 1 - y_attacked[flip_indices]

        elif self.flip_strategy == "targeted":
            # Flip specific class to another class
            target_indices = np.where(y == self.target_class)[0]
            if len(target_indices) < num_flips:
                num_flips = len(target_indices)

            flip_indices = np.random.choice(target_indices, num_flips, replace=False)
            y_attacked[flip_indices] = self.flip_to

        logger.info(f"Injected label flip attack: {num_flips} labels flipped")
        return X, y_attacked

    def get_attack_info(self) -> Dict:
        """Get attack information."""
        return {
            "attack_type": "label_flip",
            "flip_ratio": self.flip_ratio,
            "flip_strategy": self.flip_strategy,
            "target_class": self.target_class,
            "flip_to": self.flip_to,
        }
