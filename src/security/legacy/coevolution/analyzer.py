"""Co-evolution result analyzer."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .history import CoevolutionHistory, RoundMetrics

logger = logging.getLogger(__name__)


class CoevolutionAnalyzer:
    """Analyze co-evolution results."""

    def __init__(self, result: "CoevolutionResult"):
        """
        Initialize analyzer.

        Args:
            result: Co-evolution result to analyze
        """
        self.result = result
        self.history = result.history

    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends over co-evolution rounds.

        Returns:
            Dictionary with trend analysis
        """
        asr_trend = self._compute_trend(self.history.attack_success_rates)
        dr_trend = self._compute_trend(self.history.detection_rates)
        acc_trend = self._compute_trend(self.history.model_accuracies)

        return {
            "attack_success_rate_trend": asr_trend,
            "detection_rate_trend": dr_trend,
            "model_accuracy_trend": acc_trend,
        }

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return "unknown"

        # Compare first half to second half
        mid = len(values) // 2
        first_half_avg = np.mean(values[:mid])
        second_half_avg = np.mean(values[mid:])

        if second_half_avg > first_half_avg * 1.05:
            return "increasing"
        elif second_half_avg < first_half_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.history.rounds:
            return {}

        final_round = self.history.rounds[-1]

        return {
            "final_attack_success_rate": final_round.attack_success_rate,
            "final_detection_rate": final_round.detection_rate,
            "final_model_accuracy": final_round.model_accuracy,
            "total_attacker_cost": sum(
                r.attacker_cost for r in self.history.rounds
            ),
            "total_defender_cost": sum(
                r.defender_cost for r in self.history.rounds
            ),
            "avg_defense_overhead": np.mean([
                r.defense_overhead for r in self.history.rounds
            ]),
        }

    def identify_arms_race(self) -> Dict[str, Any]:
        """
        Identify if there's an arms race pattern.

        Returns:
            Dictionary with arms race analysis
        """
        if len(self.history.rounds) < 4:
            return {"arms_race": False, "reason": "Not enough rounds"}

        # Check for oscillation pattern
        asr_values = self.history.attack_success_rates
        dr_values = self.history.detection_rates

        # Count direction changes
        asr_direction_changes = 0
        dr_direction_changes = 0

        for i in range(1, len(asr_values) - 1):
            # ASR direction change
            if (asr_values[i] - asr_values[i-1]) * (asr_values[i+1] - asr_values[i]) < 0:
                asr_direction_changes += 1

            # DR direction change
            if (dr_values[i] - dr_values[i-1]) * (dr_values[i+1] - dr_values[i]) < 0:
                dr_direction_changes += 1

        # Arms race if frequent oscillations
        total_changes = asr_direction_changes + dr_direction_changes
        max_changes = 2 * (len(asr_values) - 2)

        arms_race = total_changes > max_changes * 0.3

        return {
            "arms_race": arms_race,
            "asr_direction_changes": asr_direction_changes,
            "dr_direction_changes": dr_direction_changes,
            "total_direction_changes": total_changes,
            "max_possible_changes": max_changes,
        }

    def compute_equilibrium_metrics(self) -> Dict[str, Any]:
        """Compute equilibrium-related metrics."""
        if not self.result.equilibrium_reached:
            return {
                "equilibrium_reached": False,
                "reason": "Did not converge",
            }

        equilibrium_round = self.result.equilibrium_round

        # Get metrics around equilibrium
        window = self.history.get_recent_rounds(5)

        return {
            "equilibrium_reached": True,
            "equilibrium_round": equilibrium_round,
            "equilibrium_asr": np.mean([r.attack_success_rate for r in window]),
            "equilibrium_dr": np.mean([r.detection_rate for r in window]),
            "equilibrium_accuracy": np.mean([r.model_accuracy for r in window]),
            "convergence_rate": equilibrium_round / len(self.history.rounds),
        }

    def generate_summary(self) -> str:
        """Generate text summary of co-evolution results."""
        lines = [
            "=" * 60,
            "CO-EVOLUTION SIMULATION SUMMARY",
            "=" * 60,
            "",
            f"Attack Type: {self.result.metadata.get('attack_type', 'unknown')}",
            f"Defense Type: {self.result.metadata.get('defense_type', 'unknown')}",
            f"Total Rounds: {self.result.metadata.get('num_rounds', 0)}",
            f"Total Time: {self.result.total_time:.2f}s",
            f"Equilibrium Reached: {'Yes' if self.result.equilibrium_reached else 'No'}",
            "",
            "-" * 60,
            "FINAL METRICS",
            "-" * 60,
        ]

        final_metrics = self.compute_final_metrics()
        for key, value in final_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("TRENDS")
        lines.append("-" * 60)

        trends = self.analyze_trends()
        for key, value in trends.items():
            lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("ARMS RACE ANALYSIS")
        lines.append("-" * 60)

        arms_race = self.identify_arms_race()
        for key, value in arms_race.items():
            lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("EQUILIBRIUM ANALYSIS")
        lines.append("-" * 60)

        eq_metrics = self.compute_equilibrium_metrics()
        for key, value in eq_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external analysis."""
        return {
            "final_metrics": self.compute_final_metrics(),
            "trends": self.analyze_trends(),
            "arms_race": self.identify_arms_race(),
            "equilibrium": self.compute_equilibrium_metrics(),
            "round_data": [
                {
                    "round": r.round_num,
                    "attack_success_rate": r.attack_success_rate,
                    "detection_rate": r.detection_rate,
                    "model_accuracy": r.model_accuracy,
                    "attacker_cost": r.attacker_cost,
                    "defender_cost": r.defender_cost,
                }
                for r in self.history.rounds
            ],
        }
