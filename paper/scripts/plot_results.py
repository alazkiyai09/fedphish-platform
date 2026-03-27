#!/usr/bin/env python3
"""
Plot experiment results.

Usage:
    python plot_results.py --data experiments/results/non_iid/alpha_analysis.csv
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--data", type=str, required=True, help="CSV data file")
    parser.add_argument("--output", type=str, default="figures/output", help="Output directory")
    parser.add_argument("--x", type=str, required=True, help="X-axis column")
    parser.add_argument("--y", type=str, required=True, help="Y-axis column")
    parser.add_argument("--hue", type=str, help="Grouping column")
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--xlabel", type=str, help="X-axis label")
    parser.add_argument("--ylabel", type=str, help="Y-axis label")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    if args.hue:
        for group in df[args.hue].unique():
            group_data = df[df[args.hue] == group]
            ax.plot(group_data[args.x], group_data[args.y], 'o-', label=group)
        ax.legend()
    else:
        ax.plot(df[args.x], df[args.y], 'o-')

    ax.set_xlabel(args.xlabel or args.x)
    ax.set_ylabel(args.ylabel or args.y)
    if args.title:
        ax.set_title(args.title)

    ax.grid(True, alpha=0.3)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"plot_{args.data.stem}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_file}")


if __name__ == "__main__":
    main()
