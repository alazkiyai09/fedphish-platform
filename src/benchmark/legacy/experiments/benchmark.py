"""Full benchmark orchestrator."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .runner import ExperimentRunner, ExperimentResult

logger = logging.getLogger(__name__)


class FedPhishBenchmark:
    """FedPhish Benchmark Suite orchestrator."""

    def __init__(self, config: DictConfig):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.runner = ExperimentRunner(config)

        # Results storage
        self.all_results = []
        self.summary_df = None

        # Output directories
        self.output_dir = Path(config.benchmark.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_benchmark(
        self,
        num_runs: int = 5,
    ) -> pd.DataFrame:
        """
        Run the full benchmark suite.

        Args:
            num_runs: Number of runs per configuration

        Returns:
            Summary dataframe
        """
        logger.info("Starting FedPhish Benchmark Suite")
        logger.info(f"Total configurations to test: {self._count_configurations()}")
        logger.info(f"Runs per configuration: {num_runs}")

        start_time = time.time()

        # Define experiment configurations
        experiments = self._generate_experiment_configs()

        # Run experiments
        for exp_config in experiments:
            for run_id in range(num_runs):
                logger.info(f"\n{'='*60}")
                logger.info(f"Experiment: {exp_config}")
                logger.info(f"Run {run_id + 1}/{num_runs}")
                logger.info(f"{'='*60}")

                # Run experiment
                result = self.runner.run(
                    model_type=exp_config["model_type"],
                    federation_type=exp_config["federation_type"],
                    data_distribution=exp_config["data_distribution"],
                    attack_type=exp_config["attack_type"],
                    privacy_mechanism=exp_config["privacy_mechanism"],
                    run_id=run_id,
                )

                self.all_results.append({
                    **result.config,
                    **result.metrics,
                })

        # Compile results
        self.summary_df = self._compile_results()

        # Save results
        self._save_results()

        total_time = time.time() - start_time
        logger.info(f"\nBenchmark completed in {total_time/3600:.2f} hours")

        return self.summary_df

    def _generate_experiment_configs(self) -> List[Dict]:
        """Generate all experiment configurations."""
        experiments = []

        # Model types
        model_types = self.config.benchmark.get("methods", ["xgboost"])

        # Federation types
        federation_types = self.config.benchmark.get("federations", ["fedavg"])

        # Data distributions
        distributions = self.config.benchmark.get("distributions", ["iid"])

        # Attack types
        attacks = self.config.benchmark.get("attacks", ["none"])

        # Privacy mechanisms
        privacy_mechanisms = self.config.benchmark.get("privacy", ["none"])

        # Generate all combinations
        for model_type in model_types:
            for federation_type in federation_types:
                for distribution in distributions:
                    for attack_type in attacks:
                        for privacy_mechanism in privacy_mechanisms:
                            experiments.append({
                                "model_type": model_type,
                                "federation_type": federation_type,
                                "data_distribution": distribution,
                                "attack_type": attack_type,
                                "privacy_mechanism": privacy_mechanism,
                            })

        return experiments

    def _count_configurations(self) -> int:
        """Count total number of experiment configurations."""
        experiments = self._generate_experiment_configs()
        return len(experiments) * self.config.benchmark.num_runs

    def _compile_results(self) -> pd.DataFrame:
        """Compile results into summary dataframe."""
        df = pd.DataFrame(self.all_results)

        # Group by configuration and compute statistics
        summary_columns = [
            "model_type",
            "federation_type",
            "data_distribution",
            "attack_type",
            "privacy_mechanism",
        ]

        # Metrics to aggregate
        metric_columns = [c for c in df.columns if c not in summary_columns + ["run_id", "seed", "timestamp"]]

        # Group and aggregate
        grouped = df.groupby(summary_columns)[metric_columns].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max"),
        ])

        # Flatten column names
        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()

        return grouped

    def _save_results(self) -> None:
        """Save results to files."""
        # Save detailed results
        results_path = self.output_dir / "detailed_results.csv"
        pd.DataFrame(self.all_results).to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to {results_path}")

        # Save summary
        summary_path = self.output_dir / "summary.csv"
        self.summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")

    def generate_report(self) -> None:
        """Generate LaTeX tables and figures."""
        from ..artifacts import generate_latex_tables, generate_figures

        logger.info("Generating report artifacts...")

        # Generate LaTeX tables
        generate_latex_tables(self.summary_df, self.output_dir)

        # Generate figures
        generate_figures(self.summary_df, self.output_dir)

        logger.info(f"Report artifacts saved to {self.output_dir}")


def run_full_benchmark(
    config: DictConfig,
    num_runs: int = 5,
) -> pd.DataFrame:
    """
    Convenience function to run full benchmark.

    Args:
        config: Benchmark configuration
        num_runs: Number of runs per configuration

    Returns:
        Summary dataframe
    """
    benchmark = FedPhishBenchmark(config)
    summary = benchmark.run_full_benchmark(num_runs=num_runs)
    benchmark.generate_report()
    return summary
