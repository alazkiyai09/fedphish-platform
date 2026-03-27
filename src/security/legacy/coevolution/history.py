"""Co-evolution history tracking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RoundMetrics:
    """Metrics for a single co-evolution round."""

    round_num: int
    attack_success_rate: float
    detection_rate: float
    false_positive_rate: float
    model_accuracy: float
    defense_overhead: float
    attacker_cost: float
    defender_cost: float
    attack_type: str
    defense_type: str


@dataclass
class CoevolutionHistory:
    """Complete co-evolution history."""

    rounds: List[RoundMetrics] = field(default_factory=list)
    attack_success_rates: List[float] = field(default_factory=list)
    detection_rates: List[float] = field(default_factory=list)
    model_accuracies: List[float] = field(default_factory=list)

    def add_round(self, metrics: RoundMetrics) -> None:
        """Add round metrics to history."""
        self.rounds.append(metrics)
        self.attack_success_rates.append(metrics.attack_success_rate)
        self.detection_rates.append(metrics.detection_rate)
        self.model_accuracies.append(metrics.model_accuracy)

    def get_recent_rounds(self, num_rounds: int = 5) -> List[RoundMetrics]:
        """Get recent round metrics."""
        return self.rounds[-num_rounds:]

    def check_equilibrium(
        self,
        window_size: int = 5,
        threshold: float = 0.01,
    ) -> bool:
        """
        Check if system has reached equilibrium.

        Args:
            window_size: Window size to check
            threshold: Maximum change threshold

        Returns:
            True if equilibrium reached
        """
        if len(self.rounds) < window_size:
            return False

        recent = self.get_recent_rounds(window_size)

        # Check if metrics have converged
        for metric_name in ["attack_success_rate", "detection_rate", "model_accuracy"]:
            values = [getattr(r, metric_name) for r in recent]
            if max(values) - min(values) > threshold:
                return False

        return True


# Import attack/defense history from base classes
from ..attacks.base import AttackHistory
from ..defenses.base import DefenseHistory
