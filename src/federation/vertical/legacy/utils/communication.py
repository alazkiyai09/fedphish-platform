"""Communication and logging utilities for FL experiments."""

import time
from typing import Dict, List
import json


class CommunicationTracker:
    """Track communication cost during federated training."""

    def __init__(self):
        """Initialize tracker."""
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.rounds_completed = 0

    def update_round(self, bytes_sent: int, bytes_received: int):
        """Update communication after one FL round."""
        self.total_bytes_sent += bytes_sent
        self.total_bytes_received += bytes_received
        self.rounds_completed += 1

    def get_mb_sent(self) -> float:
        """Get total MB sent."""
        return self.total_bytes_sent / (1024 * 1024)

    def get_mb_received(self) -> float:
        """Get total MB received."""
        return self.total_bytes_received / (1024 * 1024)


class ExperimentLogger:
    """Log experiment results for reproducibility."""

    def __init__(self, log_file: str = 'results/experiment_log.json'):
        """Initialize logger."""
        self.log_file = log_file
        self.experiments = []

    def log_experiment(self, config: Dict, results: Dict):
        """Log an experiment run."""
        import os
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        log_entry = {
            'timestamp': time.time(),
            'config': config,
            'results': results
        }

        self.experiments.append(log_entry)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def save_summary(self, summary_file: str = 'results/summary.json'):
        """Save experiment summary."""
        import os
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        summary = {
            'n_experiments': len(self.experiments),
            'experiments': self.experiments
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
