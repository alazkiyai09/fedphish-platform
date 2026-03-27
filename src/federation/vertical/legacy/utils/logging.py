"""Logging utilities for experiments."""

import time
import json
from typing import Dict, Any
from pathlib import Path


class ExperimentLogger:
    """Log experiment results for reproducibility."""

    def __init__(self, log_dir: str = 'results'):
        """
        Initialize logger.

        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []

    def log_experiment(self,
                     config: Dict[str, Any],
                     results: Dict[str, Any]):
        """
        Log an experiment run.

        Args:
            config: Experiment configuration
            results: Experiment results
        """
        log_entry = {
            'timestamp': time.time(),
            'config': config,
            'results': results
        }

        self.experiments.append(log_entry)

        # Save to file
        log_file = self.log_dir / f"experiment_{len(self.experiments)}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)

    def save_summary(self, summary_file: str = None):
        """
        Save summary of all experiments.

        Args:
            summary_file: Path to save summary
        """
        if summary_file is None:
            summary_file = self.log_dir / 'summary.json'

        summary = {
            'n_experiments': len(self.experiments),
            'experiments': self.experiments,
            'timestamp': time.time()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved {len(self.experiments)} experiments to {summary_file}")
