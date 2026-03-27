"""
Privacy budget tracking for federated learning.

Tracks (ε, δ) consumption across training rounds.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PrivacyBudget:
    """Privacy budget state."""
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    epsilon_total: float = 1.0
    delta_total: float = 1e-5

    def is_depleted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return (self.epsilon_spent >= self.epsilon_total or
                self.delta_spent >= self.delta_total)

    def remaining(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        eps_remaining = max(0, self.epsilon_total - self.epsilon_spent)
        delta_remaining = max(0, self.delta_total - self.delta_spent)
        return (eps_remaining, delta_remaining)


class PrivacyBudgetTracker:
    """
    Track privacy budget consumption across FL rounds.

    For DP-SGD, each round spends:
    ε = training_noise * ln(1/δ) / (batch_size * n_batches)
    """

    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        """
        Initialize privacy tracker.

        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta parameter
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta

        self.epsilon_spent = 0.0
        self.delta_spent = 0.0

        self.history: List[Tuple[int, float, float]] = []  # (round, eps, delta)

    def update(self, spent: Tuple[float, float]) -> None:
        """
        Add spent privacy budget.

        Args:
            spent: (epsilon_spent, delta_spent) tuple
        """
        epsilon_spent, delta_spent = spent

        self.epsilon_spent += epsilon_spent
        self.delta_spent += delta_spent

        self.history.append((len(self.history), epsilon_spent, delta_spent))

    def get_remaining(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.

        Returns:
            (epsilon_remaining, delta_remaining)
        """
        eps_remaining = max(0, self.total_epsilon - self.epsilon_spent)
        delta_remaining = max(0, self.total_delta - self.delta_spent)
        return (eps_remaining, delta_remaining)

    def is_depleted(self) -> bool:
        """
        Check if privacy budget is exhausted.

        Returns:
            True if budget exhausted
        """
        eps_remaining, delta_remaining = self.get_remaining()
        return eps_remaining <= 0 or delta_remaining <= 0

    def get_spent_ratio(self) -> float:
        """
        Get fraction of privacy budget spent.

        Returns:
            Fraction of ε spent (0 to 1)
        """
        return min(1.0, self.epsilon_spent / self.total_epsilon)

    def get_summary(self) -> Dict:
        """
        Get summary of privacy budget usage.

        Returns:
            Dictionary with privacy statistics
        """
        eps_remaining, delta_remaining = self.get_remaining()

        return {
            'total_epsilon': self.total_epsilon,
            'epsilon_spent': self.epsilon_spent,
            'epsilon_remaining': eps_remaining,
            'delta_spent': self.delta_spent,
            'delta_remaining': delta_remaining,
            'spent_ratio': self.get_spent_ratio(),
            'n_rounds': len(self.history),
            'is_depleted': self.is_depleted()
        }
