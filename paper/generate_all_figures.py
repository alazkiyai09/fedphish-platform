#!/usr/bin/env python3
"""
FedPhish Paper - Generate All Figures

Master script that generates all publication-quality figures.

Usage:
    python generate_all_figures.py --runs 5 --output figures/output/

Note: For quick testing, use --pre-generated flag.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate all figures for FedPhish paper")
    parser.add_argument("--runs", type=int, default=5, help="Number of experimental runs")
    parser.add_argument("--output", type=str, default="figures/output", help="Output directory")
    parser.add_argument("--pre-generated", action="store_true",
                       help="Use pre-generated results (for testing)")

    args = parser.parse_args()

    logger.info("FedPhish Paper - Figure Generation")
    logger.info("=" * 60)
    logger.info(f"Number of runs: {args.runs}")
    logger.info(f"Output directory: {args.output}")

    # Import figure generators
    from figures.src import architecture, convergence, non_iid, privacy_pareto, attacks, fairness

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    # Figure 1: System Architecture
    logger.info("\n[1/6] Figure 1: System Architecture")
    architecture.generate(output_dir)

    # Figure 2: Convergence Comparison
    logger.info("\n[2/6] Figure 2: Convergence Comparison")
    convergence.generate(output_dir)

    # Figure 3: Non-IID Impact
    logger.info("\n[3/6] Figure 3: Non-IID Impact")
    non_iid.generate(output_dir)

    # Figure 4: Privacy-Accuracy Pareto Curve
    logger.info("\n[4/6] Figure 4: Privacy-Accuracy Pareto Curve")
    privacy_pareto.generate(output_dir)

    # Figure 5: Attack Success Rate
    logger.info("\n[5/6] Figure 5: Attack Success Rate Over Rounds")
    attacks.generate(output_dir)

    # Figure 6: Per-Bank Fairness
    logger.info("\n[6/6] Figure 6: Per-Bank Fairness Analysis")
    fairness.generate(output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL FIGURES GENERATED")
    logger.info("=" * 60)
    logger.info(f"\nFigures saved to: {output_dir.absolute()}/")
    logger.info("\nGenerated files:")
    logger.info("  - fig1_architecture.pdf")
    logger.info("  - fig2_convergence.pdf")
    logger.info("  - fig3_non_iid.pdf")
    logger.info("  - fig4_privacy_pareto.pdf")
    logger.info("  - fig5_attacks.pdf")
    logger.info("  - fig6_fairness.pdf")


if __name__ == "__main__":
    main()
