"""
Visualization utilities for federated phishing detection.

Creates plots for training curves, privacy-utility tradeoffs, communication overhead, etc.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class TrainingVisualizer:
    """Visualize federated training progress."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self,
        round_metrics: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training curves (loss, accuracy per round).

        Args:
            round_metrics: List of round metric dictionaries
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not round_metrics:
            logger.warning("No round metrics to plot")
            return None

        rounds = [m["round"] for m in round_metrics]
        accuracies = [m["accuracy"] for m in round_metrics]
        losses = [m["loss"] for m in round_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(rounds, accuracies, marker="o", linewidth=2, label="Accuracy")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Validation Accuracy per Round")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Loss
        axes[1].plot(rounds, losses, marker="s", color="orange", linewidth=2, label="Loss")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Validation Loss per Round")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training curves to {path}")

        return fig

    def plot_communication_overhead(
        self,
        round_metrics: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cumulative communication overhead.

        Args:
            round_metrics: List of round metric dictionaries
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not round_metrics:
            return None

        rounds = [m["round"] for m in round_metrics]
        comm_costs = [m["communication_cost_mb"] for m in round_metrics]
        cumulative = np.cumsum(comm_costs)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(rounds, cumulative, marker="o", linewidth=2, color="green")
        ax.fill_between(rounds, cumulative, alpha=0.3, color="green")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Communication (MB)")
        ax.set_title("Cumulative Communication Overhead")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved communication overhead plot to {path}")

        return fig

    def plot_convergence_rate(
        self,
        round_metrics: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot convergence rate (parameter change over time).

        Args:
            round_metrics: List of round metric dictionaries
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not round_metrics:
            return None

        rounds = [m["round"] for m in round_metrics]
        convergence = [m.get("convergence_rate", 0) for m in round_metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(rounds, convergence, marker="o", linewidth=2, color="purple")
        ax.set_xlabel("Round")
        ax.set_ylabel("Convergence Rate (log scale)")
        ax.set_title("Model Convergence Rate")
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved convergence rate plot to {path}")

        return fig


class PrivacyVisualizer:
    """Visualize privacy metrics and tradeoffs."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_privacy_utility_curve(
        self,
        epsilons: List[float],
        accuracies: List[float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot privacy-utility tradeoff curve.

        Args:
            epsilons: List of epsilon values
            accuracies: Corresponding accuracies
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epsilons, accuracies, marker="o", linewidth=2, markersize=8)
        ax.fill_between(epsilons, accuracies, alpha=0.3)
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Privacy-Utility Tradeoff")
        ax.grid(True, alpha=0.3)

        # Mark Pareto frontier
        pareto_points = []
        max_acc = -1
        for eps, acc in sorted(zip(epsilons, accuracies)):
            if acc > max_acc:
                max_acc = acc
                pareto_points.append((eps, acc))

        if pareto_points:
            pareto_eps, pareto_accs = zip(*pareto_points)
            ax.scatter(pareto_eps, pareto_accs, color="red", s=100, zorder=5, label="Pareto Optimal")
            ax.legend()

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved privacy-utility curve to {path}")

        return fig

    def plot_privacy_cost_over_rounds(
        self,
        epsilon_history: List[float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cumulative privacy cost over rounds.

        Args:
            epsilon_history: List of epsilon values per round
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        rounds = list(range(1, len(epsilon_history) + 1))
        cumulative_eps = np.cumsum(epsilon_history)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(rounds, cumulative_eps, marker="o", linewidth=2, color="blue")
        ax.fill_between(rounds, cumulative_eps, alpha=0.3, color="blue")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative ε")
        ax.set_title("Cumulative Privacy Cost")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved privacy cost plot to {path}")

        return fig

    def compare_privacy_levels(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare different privacy levels.

        Args:
            results: Dictionary mapping privacy level -> metrics
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        levels = list(results.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Accuracy
        accuracies = [results[l].get("accuracy", 0) for l in levels]
        axes[0].bar(levels, accuracies, color="steelblue", alpha=0.8)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy by Privacy Level")
        axes[0].set_ylim([min(accuracies) * 0.9, max(accuracies) * 1.1])
        axes[0].grid(True, alpha=0.3, axis="y")

        # Communication cost
        comm_costs = [results[l].get("communication_mb", 0) for l in levels]
        axes[1].bar(levels, comm_costs, color="coral", alpha=0.8)
        axes[1].set_ylabel("Communication (MB)")
        axes[1].set_title("Communication Cost by Privacy Level")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Training time
        times = [results[l].get("training_time_sec", 0) for l in levels]
        axes[2].bar(levels, times, color="seagreen", alpha=0.8)
        axes[2].set_ylabel("Time (seconds)")
        axes[2].set_title("Training Time by Privacy Level")
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved privacy level comparison to {path}")

        return fig


class SecurityVisualizer:
    """Visualize security metrics."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_attack_defense_success(
        self,
        attack_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot attack success rates vs different defenses.

        Args:
            attack_results: Dictionary of attack -> defense -> success_rate
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        attacks = list(attack_results.keys())
        defenses = list(attack_results[attacks[0]].keys()) if attacks else []

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(attacks))
        width = 0.8 / len(defenses)

        for i, defense in enumerate(defenses):
            success_rates = [attack_results[a].get(defense, 0) for a in attacks]
            offset = (i - len(defenses) / 2 + 0.5) * width
            ax.bar(x + offset, success_rates, width, label=defense, alpha=0.8)

        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("Attack Success Rate vs Defense Strategy")
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attack defense plot to {path}")

        return fig

    def plot_reputation_scores(
        self,
        reputation_history: Dict[int, List[float]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot reputation scores over time.

        Args:
            reputation_history: Dictionary client_id -> list of scores over rounds
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for client_id, scores in reputation_history.items():
            rounds = range(1, len(scores) + 1)
            ax.plot(rounds, scores, marker="o", label=f"Client {client_id}", linewidth=2)

        ax.set_xlabel("Round")
        ax.set_ylabel("Reputation Score")
        ax.set_title("Client Reputation Scores Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved reputation scores plot to {path}")

        return fig


class BankVisualizer:
    """Visualize bank-specific metrics."""

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_bank_contributions(
        self,
        bank_metrics: Dict[int, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot contribution of each bank.

        Args:
            bank_metrics: Dictionary bank_id -> metrics
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        bank_ids = list(bank_metrics.keys())
        contributions = [bank_metrics[b].get("contribution", 0) for b in bank_ids]
        reputations = [bank_metrics[b].get("reputation", 0) for b in bank_ids]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Contribution
        axes[0].bar(bank_ids, contributions, color="steelblue", alpha=0.8)
        axes[0].set_xlabel("Bank ID")
        axes[0].set_ylabel("Contribution Score")
        axes[0].set_title("Bank Contributions")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Reputation
        axes[1].bar(bank_ids, reputations, color="coral", alpha=0.8)
        axes[1].set_xlabel("Bank ID")
        axes[1].set_ylabel("Reputation Score")
        axes[1].set_title("Bank Reputation Scores")
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            path = self.save_dir / save_path if self.save_dir else Path(save_path)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved bank contributions plot to {path}")

        return fig


def create_summary_report(
    all_metrics: Dict[str, Any],
    save_path: Optional[str] = None,
) -> str:
    """
    Create a text summary report of all metrics.

    Args:
        all_metrics: Dictionary of all metrics
        save_path: Path to save report

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEDPHISH EXPERIMENT SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Federated metrics
    if "federated" in all_metrics:
        fed = all_metrics["federated"]
        report_lines.append("FEDERATED LEARNING METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Rounds: {fed.get('total_rounds', 0)}")
        report_lines.append(f"Final Accuracy: {fed.get('final_accuracy', 0):.4f}")
        report_lines.append(f"Best Accuracy: {fed.get('best_accuracy', 0):.4f}")
        report_lines.append(f"Total Communication: {fed.get('total_communication_mb', 0):.2f} MB")
        report_lines.append(f"Total Training Time: {fed.get('total_training_time_sec', 0):.2f} sec")
        report_lines.append("")

    # Privacy metrics
    if "privacy" in all_metrics:
        priv = all_metrics["privacy"]
        report_lines.append("PRIVACY METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total ε: {priv.get('epsilon', 0):.4f}")
        report_lines.append(f"Total δ: {priv.get('delta', 0):.2e}")
        report_lines.append("")

    # Security metrics
    if "security" in all_metrics:
        sec = all_metrics["security"]
        report_lines.append("SECURITY METRICS")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Avg Proof Verification Time: {sec.get('avg_proof_verification_time', 0):.4f} sec"
        )
        report_lines.append(
            f"Avg Proof Validity Rate: {sec.get('avg_proof_validity_rate', 0):.4f}"
        )
        if "reputation" in sec and sec["reputation"]:
            rep = sec["reputation"]
            report_lines.append(f"Mean Reputation: {rep.get('mean_reputation', 0):.4f}")
        report_lines.append("")

    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(report)
        logger.info(f"Saved summary report to {path}")

    return report
