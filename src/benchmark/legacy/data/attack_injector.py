"""Attack injection for adversarial scenarios."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class AttackInjector:
    """Inject attacks into training data or model updates."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize attack injector.

        Args:
            config: Attack configuration
        """
        self.config = config

    def inject_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        attacker_id: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject attack into data.

        Args:
            X: Features
            y: Labels
            attack_type: Type of attack (label_flip, backdoor)
            attacker_id: ID of attacker client
            **kwargs: Additional attack parameters

        Returns:
            Tuple of (attacked_X, attacked_y)
        """
        if attack_type == "label_flip":
            return self._inject_label_flip(X, y, **kwargs)
        elif attack_type == "backdoor":
            return self._inject_backdoor(X, y, **kwargs)
        else:
            return X, y

    def _inject_label_flip(
        self,
        X: np.ndarray,
        y: np.ndarray,
        flip_ratio: float = 0.2,
        flip_strategy: str = "random",
        target_class: int = 1,
        flip_to: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject label flip attack.

        Args:
            X: Features
            y: Labels
            flip_ratio: Fraction of labels to flip
            flip_strategy: 'random' or 'targeted'
            target_class: Target class for targeted flipping
            flip_to: Class to flip to

        Returns:
            Tuple of (X, attacked_y)
        """
        y_attacked = y.copy()
        num_flips = int(len(y) * flip_ratio)

        if flip_strategy == "random":
            # Randomly flip labels
            flip_indices = np.random.choice(len(y), num_flips, replace=False)
            y_attacked[flip_indices] = 1 - y_attacked[flip_indices]

        elif flip_strategy == "targeted":
            # Flip specific class to another class
            target_indices = np.where(y == target_class)[0]
            if len(target_indices) < num_flips:
                num_flips = len(target_indices)

            flip_indices = np.random.choice(target_indices, num_flips, replace=False)
            y_attacked[flip_indices] = flip_to

        logger.info(f"Injected label flip attack: {num_flips} labels flipped")
        return X, y_attacked

    def _inject_backdoor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trigger_pattern: str = "https://secure-login",
        poison_ratio: float = 0.1,
        target_label: int = 1,
        source_label: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject backdoor attack into data.

        Args:
            X: Features (TF-IDF vectors, so we modify the original text)
            y: Labels
            trigger_pattern: Pattern to insert
            poison_ratio: Fraction of data to poison
            target_label: Target label (what the attacker wants)
            source_label: Source label (original label)

        Returns:
            Tuple of (attacked_X, attacked_y)
        """
        # For TF-IDF features, we need to handle this differently
        # Since we're working with pre-extracted features,
        # we'll just modify labels for now

        y_attacked = y.copy()
        source_indices = np.where(y == source_label)[0]
        num_poison = int(len(source_indices) * poison_ratio)

        if num_poison > 0:
            poison_indices = np.random.choice(source_indices, num_poison, replace=False)
            y_attacked[poison_indices] = target_label

        # Note: In a real scenario, we'd modify the text features
        # to include the trigger pattern before TF-IDF extraction

        logger.info(f"Injected backdoor attack: {num_poison} samples poisoned")
        return X, y_attacked


def inject_label_flip(
    X: np.ndarray,
    y: np.ndarray,
    flip_ratio: float = 0.2,
    flip_strategy: str = "random"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to inject label flip attack.

    Args:
        X: Features
        y: Labels
        flip_ratio: Fraction of labels to flip
        flip_strategy: 'random' or 'targeted'

    Returns:
        Tuple of (X, attacked_y)
    """
    injector = AttackInjector()
    return injector._inject_label_flip(X, y, flip_ratio=flip_ratio, flip_strategy=flip_strategy)


def inject_backdoor(
    X: np.ndarray,
    y: np.ndarray,
    trigger_pattern: str = "https://secure-login",
    poison_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to inject backdoor attack.

    Args:
        X: Features
        y: Labels
        trigger_pattern: Trigger pattern
        poison_ratio: Fraction of data to poison

    Returns:
        Tuple of (attacked_X, attacked_y)
    """
    injector = AttackInjector()
    return injector._inject_backdoor(X, y, trigger_pattern=trigger_pattern, poison_ratio=poison_ratio)


def scale_gradients(
    original_gradients: List[np.ndarray],
    scaling_factor: float
) -> List[np.ndarray]:
    """
    Scale gradients for model poisoning attack.

    Args:
        original_gradients: List of gradient arrays
        scaling_factor: Scaling factor (negative for degradation)

    Returns:
        List of scaled gradients
    """
    scaled_gradients = [g * scaling_factor for g in original_gradients]
    logger.info(f"Scaled gradients by factor {scaling_factor}")
    return scaled_gradients


def inject_backdoor_updates(
    original_updates: List[np.ndarray],
    trigger_pattern: np.ndarray,
    target_label: int
) -> List[np.ndarray]:
    """
    Inject backdoor into model updates.

    Args:
        original_updates: Original model updates
        trigger_pattern: Trigger pattern
        target_label: Target label for backdoor

    Returns:
        List of poisoned updates
    """
    # This is a simplified version
    # In practice, you'd modify the updates to embed the backdoor
    poisoned_updates = original_updates.copy()

    # Add trigger pattern to first layer weights
    if len(poisoned_updates) > 0:
        # Modify first layer to recognize trigger
        poisoned_updates[0] += trigger_pattern * 0.1

    logger.info("Injected backdoor into model updates")
    return poisoned_updates
