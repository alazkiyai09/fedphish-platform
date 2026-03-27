"""Visualization for co-evolution results."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..coevolution.history import CoevolutionHistory

logger = logging.getLogger(__name__)


class CoevolutionVisualizer:
    """Visualize co-evolution dynamics."""

    def __init__(
        self,
        history: CoevolutionHistory,
        output_dir: str = "./results/plots",
    ):
        """
        Initialize visualizer.

        Args:
            history: Co-evolution history
            output_dir: Output directory for plots
        """
        self.history = history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_over_rounds(
        self,
        metrics: List[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot metrics over co-evolution rounds.

        Args:
            metrics: List of metrics to plot
            save: Whether to save plot
            show: Whether to display plot
        """
        if metrics is None:
            metrics = ["attack_success_rate", "detection_rate", "model_accuracy"]

        rounds = [r.round_num for r in self.history.rounds]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [getattr(r, metric) for r in self.history.rounds]

            ax.plot(rounds, values, marker='o', linewidth=2)
            ax.set_xlabel("Round")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} over Rounds")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "metrics_over_rounds.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_costs_over_rounds(
        self,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot attacker and defender costs over rounds.

        Args:
            save: Whether to save plot
            show: Whether to display plot
        """
        rounds = [r.round_num for r in self.history.rounds]
        attacker_costs = [r.attacker_cost for r in self.history.rounds]
        defender_costs = [r.defender_cost for r in self.history.rounds]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(rounds, attacker_costs, marker='o', label="Attacker Cost", linewidth=2)
        ax.plot(rounds, defender_costs, marker='s', label="Defender Cost", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cost")
        ax.set_title("Costs over Co-evolution Rounds")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "costs_over_rounds.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_attack_defense_dynamics(
        self,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot attack-defense dynamics (success rate vs detection rate).

        Args:
            save: Whether to save plot
            show: Whether to display plot
        """
        asr_values = [r.attack_success_rate for r in self.history.rounds]
        dr_values = [r.detection_rate for r in self.history.rounds]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot trajectory
        ax.plot(asr_values, dr_values, marker='o', linewidth=2, label="Trajectory")

        # Annotate rounds
        for i, (asr, dr) in enumerate(zip(asr_values, dr_values)):
            ax.annotate(
                str(i + 1),
                (asr, dr),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("Attack Success Rate")
        ax.set_ylabel("Detection Rate")
        ax.set_title("Attack-Defense Dynamics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "attack_defense_dynamics.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_equilibrium_analysis(
        self,
        window_size: int = 5,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot equilibrium analysis.

        Args:
            window_size: Window size for moving average
            save: Whether to save plot
            show: Whether to display plot
        """
        rounds = [r.round_num for r in self.history.rounds]
        asr_values = [r.attack_success_rate for r in self.history.rounds]
        dr_values = [r.detection_rate for r in self.history.rounds]

        # Compute moving averages
        def moving_average(values, window):
            if len(values) < window:
                return values
            return [
                np.mean(values[max(0, i - window):i + 1])
                for i in range(len(values))
            ]

        asr_ma = moving_average(asr_values, window_size)
        dr_ma = moving_average(dr_values, window_size)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Attack success rate
        axes[0].plot(rounds, asr_values, alpha=0.5, label="Raw")
        axes[0].plot(rounds, asr_ma, linewidth=2, label=f"MA({window_size})")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Attack Success Rate")
        axes[0].set_title("Attack Success Rate - Equilibrium Analysis")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Detection rate
        axes[1].plot(rounds, dr_values, alpha=0.5, label="Raw")
        axes[1].plot(rounds, dr_ma, linewidth=2, label=f"MA({window_size})")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Detection Rate")
        axes[1].set_title("Detection Rate - Equilibrium Analysis")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "equilibrium_analysis.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_arms_race_heatmap(
        self,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot arms race as heatmap.

        Args:
            save: Whether to save plot
            show: Whether to display plot
        """
        asr_values = [r.attack_success_rate for r in self.history.rounds]
        dr_values = [r.detection_rate for r in self.history.rounds]

        # Create 2D histogram
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot density
        hist = ax.hist2d(
            asr_values,
            dr_values,
            bins=20,
            cmap="YlOrRd",
        )

        ax.set_xlabel("Attack Success Rate")
        ax.set_ylabel("Detection Rate")
        ax.set_title("Arms Race Heatmap")

        plt.colorbar(hist[3], ax=ax, label="Frequency")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "arms_race_heatmap.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()
