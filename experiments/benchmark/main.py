#!/usr/bin/env python
"""Main entry point for FedPhish Benchmark Suite."""

import sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from src.experiments import run_full_benchmark, run_single_experiment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FedPhish Benchmark Suite - Federated Phishing Detection Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark.yaml",
        help="Path to benchmark configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "single"],
        default="full",
        help="Benchmark mode: full benchmark or single experiment",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Model type (for single mode)",
    )

    parser.add_argument(
        "--federation",
        type=str,
        default="fedavg",
        help="Federation type (for single mode)",
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default="iid",
        help="Data distribution (for single mode)",
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        help="Attack type (for single mode)",
    )

    parser.add_argument(
        "--privacy",
        type=str,
        default="none",
        help="Privacy mechanism (for single mode)",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per configuration",
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run ID (for single mode)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # Override output directory if specified
    if args.output_dir:
        config.benchmark.output_dir = args.output_dir

    print("=" * 60)
    print("FedPhish Benchmark Suite")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Output directory: {config.benchmark.output_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    print()

    # Run benchmark
    if args.mode == "full":
        print("Running full benchmark suite...")
        print(f"Configurations: {len(config.benchmark.methods) * len(config.benchmark.federations)}")
        print(f"Runs per configuration: {args.num_runs}")
        print()

        results = run_full_benchmark(config, num_runs=args.num_runs)

        print()
        print("=" * 60)
        print("Benchmark completed!")
        print(f"Results saved to: {config.benchmark.output_dir}")
        print("=" * 60)

    elif args.mode == "single":
        print("Running single experiment...")
        print(f"Model: {args.model}")
        print(f"Federation: {args.federation}")
        print(f"Data distribution: {args.distribution}")
        print(f"Attack: {args.attack}")
        print(f"Privacy: {args.privacy}")
        print(f"Run ID: {args.run_id}")
        print()

        result = run_single_experiment(
            config=config,
            model_type=args.model,
            federation_type=args.federation,
            data_distribution=args.distribution,
            attack_type=args.attack,
            privacy_mechanism=args.privacy,
            run_id=args.run_id,
        )

        print()
        print("=" * 60)
        print("Experiment completed!")
        print(f"Accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
        print(f"AUPRC: {result.metrics.get('auprc', 'N/A'):.4f}")
        print(f"Training time: {result.training_time:.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
