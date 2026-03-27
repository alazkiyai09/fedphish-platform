#!/usr/bin/env python3
"""
FedPhish Paper - Full Evaluation Pipeline

Master script that:
1. Runs all experiments (5 runs, 95% CI)
2. Generates all figures (publication quality)
3. Generates all tables (LaTeX format)
4. Outputs to paper/ and supplementary/

Usage:
    python run_full_evaluation.py --runs 5 --quick-test
    python run_full_evaluation.py --runs 5 --full-eval

For quick testing without running experiments, use --quick-test.
For full evaluation with GPU experiments, use --full-eval.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiments(runs: int, quick_test: bool = False):
    """Run all experiments."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RUNNING EXPERIMENTS")
    logger.info("=" * 60)

    if quick_test:
        logger.info("Quick test mode - skipping experiments, using placeholders")
        return

    # Experiment configurations
    configs = [
        "experiments/configs/detection_comparison.yaml",
        "experiments/configs/privacy_utility.yaml",
        "experiments/configs/robustness.yaml",
        "experiments/configs/overhead.yaml",
        "experiments/configs/non_iid.yaml",
    ]

    for config in configs:
        logger.info(f"\nRunning experiment: {config}")
        cmd = [
            "python3", "scripts/run_single_exp.py",
            "--config", config,
            "--runs", str(runs),
            "--output", f"experiments/results/{Path(config).stem}/"
        ]
        subprocess.run(cmd, check=True)


def generate_figures(output_dir: str = "figures/output"):
    """Generate all figures."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: GENERATING FIGURES")
    logger.info("=" * 60)

    from figures.src import architecture, convergence, non_iid, privacy_pareto, attacks, fairness

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Figure 1: Architecture
    logger.info("\n[1/6] Figure 1: System Architecture")
    architecture.generate(output_path)

    # Figure 2: Convergence
    logger.info("\n[2/6] Figure 2: Convergence Comparison")
    convergence.generate(output_path)

    # Figure 3: Non-IID
    logger.info("\n[3/6] Figure 3: Non-IID Impact")
    non_iid.generate(output_path)

    # Figure 4: Privacy Pareto
    logger.info("\n[4/6] Figure 4: Privacy-Accuracy Pareto Frontier")
    privacy_pareto.generate(output_path)

    # Figure 5: Attacks
    logger.info("\n[5/6] Figure 5: Attack Success Rate")
    attacks.generate(output_path)

    # Figure 6: Fairness
    logger.info("\n[6/6] Figure 6: Per-Bank Fairness")
    fairness.generate(output_path)


def generate_tables(output_dir: str = "tables/output"):
    """Generate all tables."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: GENERATING TABLES")
    logger.info("=" * 60)

    from tables.src import detection_performance, privacy_utility, robustness, overhead

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table 1: Detection Performance
    logger.info("\n[1/4] Table 1: Detection Performance Comparison")
    detection_performance.generate()

    # Table 2: Privacy-Utility
    logger.info("\n[2/4] Table 2: Privacy-Utility Trade-off")
    privacy_utility.generate()

    # Table 3: Robustness
    logger.info("\n[3/4] Table 3: Robustness Against Attacks")
    robustness.generate()

    # Table 4: Overhead
    logger.info("\n[4/4] Table 4: Overhead Analysis")
    overhead.generate()


def verify_outputs():
    """Verify all outputs were generated."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: VERIFYING OUTPUTS")
    logger.info("=" * 60)

    # Check figures
    figures_dir = Path("figures/output")
    required_figures = [
        "fig1_architecture.pdf",
        "fig2_convergence.pdf",
        "fig3_non_iid.pdf",
        "fig4_privacy_pareto.pdf",
        "fig5_attacks.pdf",
        "fig6_fairness.pdf",
    ]

    logger.info("\nChecking figures...")
    for fig in required_figures:
        fig_path = figures_dir / fig
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            logger.info(f"  ✅ {fig} ({size_kb:.1f} KB)")
        else:
            logger.warning(f"  ❌ {fig} missing")

    # Check tables
    tables_dir = Path("tables/output")
    required_tables = [
        "table1_detection.tex",
        "table2_privacy.tex",
        "table3_robustness.tex",
        "table4_overhead.tex",
    ]

    logger.info("\nChecking tables...")
    for tab in required_tables:
        tab_path = tables_dir / tab
        if tab_path.exists():
            logger.info(f"  ✅ {tab}")
        else:
            logger.warning(f"  ❌ {tab} missing")


def main():
    parser = argparse.ArgumentParser(
        description="Run full FedPhish evaluation pipeline"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of experimental runs for statistical significance"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Base output directory"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: generate figures/tables without running experiments"
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiments, use existing results"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation"
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip table generation"
    )

    args = parser.parse_args()

    logger.info("FedPhish Paper - Full Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Number of runs: {args.runs}")
    logger.info(f"Quick test mode: {args.quick_test}")

    try:
        # Phase 1: Experiments
        if not args.skip_experiments:
            run_experiments(args.runs, args.quick_test)

        # Phase 2: Figures
        if not args.skip_figures:
            generate_figures(f"{args.output}/figures/output")

        # Phase 3: Tables
        if not args.skip_tables:
            generate_tables(f"{args.output}/tables/output")

        # Phase 4: Verification
        verify_outputs()

        logger.info("\n" + "=" * 60)
        logger.info("✅ FULL EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nAll outputs saved to: {Path(args.output).absolute()}/")
        logger.info("\nGenerated files:")
        logger.info("  Figures:")
        logger.info("    - figures/output/fig1_architecture.pdf")
        logger.info("    - figures/output/fig2_convergence.pdf")
        logger.info("    - figures/output/fig3_non_iid.pdf")
        logger.info("    - figures/output/fig4_privacy_pareto.pdf")
        logger.info("    - figures/output/fig5_attacks.pdf")
        logger.info("    - figures/output/fig6_fairness.pdf")
        logger.info("  Tables:")
        logger.info("    - tables/output/table1_detection.tex")
        logger.info("    - tables/output/table2_privacy.tex")
        logger.info("    - tables/output/table3_robustness.tex")
        logger.info("    - tables/output/table4_overhead.tex")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
