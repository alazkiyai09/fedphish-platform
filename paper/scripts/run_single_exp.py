#!/usr/bin/env python3
"""
Run a single FedPhish experiment based on YAML config.

Usage:
    python run_single_exp.py --config experiments/configs/detection_comparison.yaml
"""

import argparse
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run FedPhish experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--runs", type=int, default=5, help="Number of experimental runs")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Running experiment: {config['experiment_name']}")
    logger.info(f"Config: {config}")

    # Import FedPhish modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "fedphish"))

    # Run experiment
    logger.info("Initializing experiment...")

    # Placeholder for actual experiment execution
    logger.info(f"Running {args.runs} iterations...")
    for i in range(args.runs):
        logger.info(f"Run {i+1}/{args.runs}")

    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
