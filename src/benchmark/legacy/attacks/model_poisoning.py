"""Model poisoning attack implementation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ModelPoisoningAttack:
    """Model poisoning attack (gradient scaling, sign flip)."""

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize model poisoning attack.

        Args:
            config: Attack configuration
        """
        self.config = config or {}
        attack_config = self.config.get("model_poisoning", {})

        self.attack_type = attack_config.get("attack_type", "scale_gradients")
        self.scaling_factor = attack_config.get("scaling_factor", -5.0)
        self.scale_all_layers = attack_config.get("scale_all_layers", True)
        self.sign_flip_ratio = attack_config.get("sign_flip_ratio", 1.0)

    def inject(
        self,
        parameters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Inject model poisoning attack.

        Args:
            parameters: Model parameters/gradients

        Returns:
            Poisoned parameters
        """
        poisoned_params = parameters.copy()

        if self.attack_type == "scale_gradients":
            # Scale gradients
            poisoned_params = self._scale_gradients(poisoned_params)

        elif self.attack_type == "sign_flip":
            # Flip gradient signs
            poisoned_params = self._flip_signs(poisoned_params)

        elif self.attack_type == "byzantine":
            # Byzantine attack: send random updates
            poisoned_params = self._byzantine_attack(poisoned_params)

        logger.info(f"Injected {self.attack_type} attack")
        return poisoned_params

    def _scale_gradients(
        self,
        parameters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Scale gradients."""
        if self.scale_all_layers:
            return [p * self.scaling_factor for p in parameters]
        else:
            # Only scale first layer
            scaled = parameters.copy()
            scaled[0] = scaled[0] * self.scaling_factor
            return scaled

    def _flip_signs(
        self,
        parameters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Flip gradient signs."""
        return [-p * self.sign_flip_ratio for p in parameters]

    def _byzantine_attack(
        self,
        parameters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Generate random Byzantine updates."""
        return [np.random.randn(*p.shape) for p in parameters]

    def evaluate_attack_success(
        self,
        global_model,
        clean_parameters: List[np.ndarray],
        poisoned_parameters: List[np.ndarray],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate attack success.

        Args:
            global_model: Global model
            clean_parameters: Clean model parameters
            poisoned_parameters: Poisoned model parameters
            X_test: Test features
            y_test: Test labels

        Returns:
            Attack metrics
        """
        # Set clean parameters and evaluate
        global_model.set_parameters(clean_parameters)
        clean_metrics = global_model.evaluate(X_test, y_test)

        # Set poisoned parameters and evaluate
        global_model.set_parameters(poisoned_parameters)
        poisoned_metrics = global_model.evaluate(X_test, y_test)

        # Calculate degradation
        accuracy_degradation = clean_metrics["accuracy"] - poisoned_metrics["accuracy"]

        return {
            "accuracy_degradation": accuracy_degradation,
            "clean_accuracy": clean_metrics["accuracy"],
            "poisoned_accuracy": poisoned_metrics["accuracy"],
            "attack_success_rate": min(accuracy_degradation / clean_metrics["accuracy"], 1.0),
        }

    def get_attack_info(self) -> Dict:
        """Get attack information."""
        return {
            "attack_type": "model_poisoning",
            "sub_type": self.attack_type,
            "scaling_factor": self.scaling_factor,
        }
