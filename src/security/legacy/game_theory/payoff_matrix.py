"""Payoff matrix computation for attack-defense game."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .utility import AttackerUtility, DefenderUtility

logger = logging.getLogger(__name__)


class PayoffMatrix:
    """Payoff matrix for attacker-defender game."""

    def __init__(
        self,
        attack_actions: List[str],
        defense_actions: List[str],
        attacker_utility: AttackerUtility = None,
        defender_utility: DefenderUtility = None,
    ):
        """
        Initialize payoff matrix.

        Args:
            attack_actions: List of attack strategies
            defense_actions: List of defense strategies
            attacker_utility: Attacker utility function
            defender_utility: Defender utility function
        """
        self.attack_actions = attack_actions
        self.defense_actions = defense_actions
        self.attacker_utility = attacker_utility or AttackerUtility()
        self.defender_utility = defender_utility or DefenderUtility()

        # Payoff matrices: [attack_idx][defense_idx]
        self.attacker_payoffs = np.zeros((len(attack_actions), len(defense_actions)))
        self.defender_payoffs = np.zeros((len(attack_actions), len(defense_actions)))

        self._computed = False

    def compute_payoffs(
        self,
        outcomes: Dict[Tuple[str, str], Dict[str, float]],
    ) -> None:
        """
        Compute payoff matrix from simulation outcomes.

        Args:
            outcomes: Dictionary mapping (attack, defense) to outcome metrics
                Outcome metrics should include:
                - attack_success: bool
                - evasion_success: bool
                - detection_rate: float
                - false_positive_rate: float
                - model_accuracy: float
                - attacker_cost: float
                - defender_cost: float
        """
        for i, attack in enumerate(self.attack_actions):
            for j, defense in enumerate(self.defense_actions):
                outcome = outcomes.get((attack, defense), {})

                # Attacker payoff
                attacker_utility = self.attacker_utility.compute(
                    attack_success=outcome.get("attack_success", False),
                    evasion_success=outcome.get("evasion_success", False),
                    cost=outcome.get("attacker_cost", 0.0),
                    model_accuracy=outcome.get("model_accuracy", 1.0),
                    exposure_risk=outcome.get("exposure_risk", 0.0),
                )

                # Defender payoff
                defender_utility = self.defender_utility.compute(
                    detection_rate=outcome.get("detection_rate", 0.0),
                    false_positive_rate=outcome.get("false_positive_rate", 0.0),
                    model_accuracy=outcome.get("model_accuracy", 1.0),
                    cost=outcome.get("defender_cost", 0.0),
                )

                self.attacker_payoffs[i, j] = attacker_utility
                self.defender_payoffs[i, j] = defender_utility

        self._computed = True

        logger.info("Payoff matrix computed")

    def get_attacker_payoff(self, attack_idx: int, defense_idx: int) -> float:
        """Get attacker payoff for specific action pair."""
        return self.attacker_payoffs[attack_idx, defense_idx]

    def get_defender_payoff(self, attack_idx: int, defense_idx: int) -> float:
        """Get defender payoff for specific action pair."""
        return self.defender_payoffs[attack_idx, defense_idx]

    def get_attacker_best_response(self, defense_idx: int) -> int:
        """Get attacker's best response to specific defense."""
        return int(np.argmax(self.attacker_payoffs[:, defense_idx]))

    def get_defender_best_response(self, attack_idx: int) -> int:
        """Get defender's best response to specific attack."""
        return int(np.argmax(self.defender_payoffs[attack_idx, :]))

    def get_payoff_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get both payoff matrices."""
        return self.attacker_payoffs, self.defender_payoffs

    def print_matrix(self) -> None:
        """Print payoff matrix in readable format."""
        if not self._computed:
            logger.warning("Payoff matrix not computed yet")
            return

        print("\n" + "=" * 60)
        print("PAYOFF MATRIX (Attacker, Defender)")
        print("=" * 60)
        print(f"{'Attack':<20}", end="")
        for defense in self.defense_actions:
            print(f"{defense:<15}", end="")
        print()
        print("-" * 60)

        for i, attack in enumerate(self.attack_actions):
            print(f"{attack:<20}", end="")
            for j in range(len(self.defense_actions)):
                attacker_payoff = self.attacker_payoffs[i, j]
                defender_payoff = self.defender_payoffs[i, j]
                print(f"({attacker_payoff:5.2f}, {defender_payoff:5.2f})", end=" ")
            print()

        print("=" * 60 + "\n")
