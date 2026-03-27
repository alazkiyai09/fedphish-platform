"""Utility functions for attacker and defender."""

import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AttackerUtility:
    """Attacker utility function."""

    # Weight parameters
    success_weight: float = 1.0
    evasion_weight: float = 0.5
    accuracy_degradation_weight: float = 0.3
    cost_weight: float = 0.1
    exposure_weight: float = 0.2

    def compute(
        self,
        attack_success: bool,
        evasion_success: bool,
        cost: float,
        model_accuracy: float,
        exposure_risk: float = 0.0,
    ) -> float:
        """
        Compute attacker utility.

        U_attacker = success_benefit - cost - exposure_penalty

        Args:
            attack_success: Whether attack succeeded
            evasion_success: Whether evasion succeeded
            cost: Attacker cost
            model_accuracy: Final model accuracy (lower = better for attacker)
            exposure_risk: Risk of being exposed

        Returns:
            Utility value
        """
        # Success benefit
        success_benefit = 0.0
        if attack_success:
            success_benefit += self.success_weight

        if evasion_success:
            success_benefit += self.evasion_weight

        # Accuracy degradation (attacker wants low accuracy)
        accuracy_degradation = (1.0 - model_accuracy) * self.accuracy_degradation_weight
        success_benefit += accuracy_degradation

        # Cost penalty
        cost_penalty = cost * self.cost_weight

        # Exposure penalty
        exposure_penalty = exposure_risk * self.exposure_weight

        utility = success_benefit - cost_penalty - exposure_penalty

        logger.debug(
            f"Attacker utility: success={success_benefit:.3f}, "
            f"cost={cost_penalty:.3f}, exposure={exposure_penalty:.3f}, "
            f"total={utility:.3f}"
        )

        return utility


@dataclass
class DefenderUtility:
    """Defender utility function."""

    # Weight parameters
    accuracy_weight: float = 1.0
    detection_weight: float = 0.5
    cost_weight: float = 0.1
    fp_penalty_weight: float = 0.3

    def compute(
        self,
        detection_rate: float,
        false_positive_rate: float,
        model_accuracy: float,
        cost: float,
    ) -> float:
        """
        Compute defender utility.

        U_defender = accuracy_benefit + detection_benefit - cost - fp_penalty

        Args:
            detection_rate: Rate of detecting attacks
            false_positive_rate: Rate of false positives
            model_accuracy: Final model accuracy
            cost: Defender cost

        Returns:
            Utility value
        """
        # Accuracy benefit
        accuracy_benefit = model_accuracy * self.accuracy_weight

        # Detection benefit
        detection_benefit = detection_rate * self.detection_weight

        # Cost penalty
        cost_penalty = cost * self.cost_weight

        # False positive penalty
        fp_penalty = false_positive_rate * self.fp_penalty_weight

        utility = accuracy_benefit + detection_benefit - cost_penalty - fp_penalty

        logger.debug(
            f"Defender utility: accuracy={accuracy_benefit:.3f}, "
            f"detection={detection_benefit:.3f}, cost={cost_penalty:.3f}, "
            f"fp={fp_penalty:.3f}, total={utility:.3f}"
        )

        return utility
