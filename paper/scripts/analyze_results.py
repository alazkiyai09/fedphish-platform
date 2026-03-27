#!/usr/bin/env python3
"""
Analyze experiment results and generate statistics.

Usage:
    python analyze_results.py --results experiments/results/non_iid/ --output analysis/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV results from directory."""
    results = []
    for file in results_dir.glob("*.csv"):
        df = pd.read_csv(file)
        results.append(df)
    return pd.concat(results, ignore_index=True)


def compute_ci(data: np.ndarray, confidence=0.95) -> tuple:
    """Compute confidence interval."""
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return mean, ci[1] - mean


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results", type=str, required=True, help="Results directory")
    parser.add_argument("--output", type=str, default="analysis", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")

    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    df = load_results(results_dir)

    print(f"\nLoaded {len(df)} results")
    print("\nSummary statistics:")
    print(df.describe())

    # Compute confidence intervals
    print("\nConfidence Intervals:")
    for column in df.select_dtypes(include=[np.number]).columns:
        mean, ci = compute_ci(df[column].dropna(), args.confidence)
        print(f"{column}: {mean:.3f} ± {ci:.3f}")

    # Save analysis
    df.to_csv(output_dir / "analysis_summary.csv", index=False)

    # Save statistics
    stats_dict = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        mean, ci = compute_ci(df[column].dropna(), args.confidence)
        stats_dict[column] = {
            "mean": float(mean),
            "ci": float(ci),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
        }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\n✅ Analysis saved to {output_dir}/")


if __name__ == "__main__":
    main()
