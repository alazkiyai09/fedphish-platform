"""Nash equilibrium analysis."""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import linprog

from .payoff_matrix import PayoffMatrix

logger = logging.getLogger(__name__)


class NashEquilibriumAnalyzer:
    """Analyze Nash equilibrium in attack-defense game."""

    def __init__(self, payoff_matrix: PayoffMatrix):
        """
        Initialize analyzer.

        Args:
            payoff_matrix: Payoff matrix
        """
        self.payoff_matrix = payoff_matrix

    def find_pure_nash_equilibrium(self) -> List[Tuple[int, int]]:
        """
        Find pure strategy Nash equilibria.

        Returns:
            List of (attack_idx, defense_idx) tuples that are Nash equilibria
        """
        attacker_payoffs, defender_payoffs = self.payoff_matrix.get_payoff_matrix()

        n_attacks, n_defenses = attacker_payoffs.shape

        equilibria = []

        for i in range(n_attacks):
            for j in range(n_defenses):
                # Check if (i, j) is a Nash equilibrium
                # Attacker's best response to defense j
                attacker_best = self.payoff_matrix.get_attacker_best_response(j)

                # Defender's best response to attack i
                defender_best = self.payoff_matrix.get_defender_best_response(i)

                # Nash equilibrium: both are playing best response
                if i == attacker_best and j == defender_best:
                    equilibria.append((i, j))

        if equilibria:
            logger.info(f"Found {len(equilibria)} pure Nash equilibria")
        else:
            logger.info("No pure Nash equilibrium found")

        return equilibria

    def find_mixed_nash_equilibrium(
        self,
        method: str = "support_enumeration",
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Find mixed strategy Nash equilibrium.

        Args:
            method: Method to use ("support_enumeration", "lemke_howson")

        Returns:
            Dictionary with 'attacker_strategy' and 'defender_strategy' arrays,
            or None if not found
        """
        if method == "support_enumeration":
            return self._support_enumeration()
        else:
            logger.warning(f"Unknown method: {method}")
            return None

    def _support_enumeration(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Support enumeration algorithm for finding mixed Nash equilibrium.

        Returns:
            Dictionary with mixed strategies, or None
        """
        attacker_payoffs, defender_payoffs = self.payoff_matrix.get_payoff_matrix()
        n_attacks, n_defenses = attacker_payoffs.shape

        # Try all possible supports
        from itertools import combinations

        # Attacker support size k, defender support size l
        # where k * l >= n_attacks + n_defenses - 1

        for k in range(1, n_attacks + 1):
            for l in range(1, n_defenses + 1):
                if k * l < n_attacks + n_defenses - 1:
                    continue

                # Try all supports of size k and l
                for attack_support in combinations(range(n_attacks), k):
                    for defense_support in combinations(range(n_defenses), l):
                        result = self._solve_equilibrium_on_supports(
                            list(attack_support),
                            list(defense_support),
                            attacker_payoffs,
                            defender_payoffs,
                        )

                        if result is not None:
                            logger.info("Found mixed Nash equilibrium via support enumeration")
                            return result

        logger.info("No mixed Nash equilibrium found")
        return None

    def _solve_equilibrium_on_supports(
        self,
        attack_support: List[int],
        defense_support: List[int],
        attacker_payoffs: np.ndarray,
        defender_payoffs: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Solve for equilibrium on given supports."""
        import warnings
        warnings.filterwarnings('ignore')

        k = len(attack_support)
        l = len(defense_support)

        # Extract submatrices
        A_sub = attacker_payoffs[np.ix_(attack_support, defense_support)]
        B_sub = defender_payoffs[np.ix_(attack_support, defense_support)]

        # Solve for mixed strategies
        try:
            # Defender's strategy: makes attacker indifferent
            # Solve A * y = 1 * v
            A_def = np.vstack([A_sub.T, np.ones(k)])
            b_def = np.hstack([np.zeros(l), [1]])

            result_def = np.linalg.lstsq(A_def, b_def, rcond=None)[0]
            y = result_def[:l]

            # Attacker's strategy: makes defender indifferent
            # Solve x^T * B = 1 * v
            A_att = np.vstack([B_sub, np.ones(l)])
            b_att = np.hstack([np.zeros(k), [1]])

            result_att = np.linalg.lstsq(A_att.T, b_att, rcond=None)[0]
            x = result_att[:k]

            # Check if valid (all probabilities >= 0)
            if np.all(x >= -1e-6) and np.all(y >= -1e-6):
                # Normalize
                x = np.maximum(x, 0)
                y = np.maximum(y, 0)
                x = x / x.sum()
                y = y / y.sum()

                # Expand to full strategies
                full_x = np.zeros(len(attacker_payoffs))
                full_y = np.zeros(len(defender_payoffs))

                full_x[list(attack_support)] = x
                full_y[list(defense_support)] = y

                return {
                    "attacker_strategy": full_x,
                    "defender_strategy": full_y,
                }

        except Exception as e:
            return None

        return None

    def compute_dominant_strategies(self) -> Dict[str, Optional[int]]:
        """
        Compute dominant strategies for both players.

        Returns:
            Dictionary with 'attacker_dominant' and 'defender_dominant' indices,
            or None if no dominant strategy exists
        """
        attacker_payoffs, defender_payoffs = self.payoff_matrix.get_payoff_matrix()

        # Check for attacker's dominant strategy
        attacker_dominant = None
        for i in range(len(attacker_payoffs)):
            is_dominant = True
            for j in range(len(attacker_payoffs)):
                if i == j:
                    continue
                if not np.all(attacker_payoffs[i] >= attacker_payoffs[j]):
                    is_dominant = False
                    break

            if is_dominant:
                attacker_dominant = i
                break

        # Check for defender's dominant strategy
        defender_dominant = None
        for j in range(defender_payoffs.shape[1]):
            is_dominant = True
            for k in range(defender_payoffs.shape[1]):
                if j == k:
                    continue
                if not np.all(defender_payoffs[:, j] >= defender_payoffs[:, k]):
                    is_dominant = False
                    break

            if is_dominant:
                defender_dominant = j
                break

        result = {
            "attacker_dominant": attacker_dominant,
            "defender_dominant": defender_dominant,
        }

        if attacker_dominant is not None:
            logger.info(f"Attacker has dominant strategy: {attacker_dominant}")
        if defender_dominant is not None:
            logger.info(f"Defender has dominant strategy: {defender_dominant}")

        return result

    def analyze_pareto_optimal(self) -> List[Tuple[int, int]]:
        """
        Find Pareto-optimal outcomes.

        Returns:
            List of (attack_idx, defense_idx) tuples
        """
        attacker_payoffs, defender_payoffs = self.payoff_matrix.get_payoff_matrix()
        n_attacks, n_defenses = attacker_payoffs.shape

        pareto_optimal = []

        for i in range(n_attacks):
            for j in range(n_defenses):
                is_pareto = True

                # Check if any other outcome dominates this one
                for ii in range(n_attacks):
                    for jj in range(n_defenses):
                        if ii == i and jj == j:
                            continue

                        # (ii, jj) dominates (i, j) if both players are better off
                        if (attacker_payoffs[ii, jj] > attacker_payoffs[i, j] and
                            defender_payoffs[ii, jj] > defender_payoffs[i, j]):
                            is_pareto = False
                            break

                    if not is_pareto:
                        break

                if is_pareto:
                    pareto_optimal.append((i, j))

        logger.info(f"Found {len(pareto_optimal)} Pareto-optimal outcomes")

        return pareto_optimal
