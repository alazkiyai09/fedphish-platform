#!/usr/bin/env python3
"""Main entry point for co-evolution experiments."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.attacks import (
    DefenseAwareLabelFlip,
    DefenseAwareBackdoor,
    DefenseAwareModelPoisoning,
    EvasionPoisoningCombo,
    AttackerKnowledge,
    AttackConfig,
)
from src.defenses import (
    MultiRoundAnomalyDetection,
    HoneypotDefense,
    GradientForensics,
    DefenderObservability,
    DefenseConfig,
)
from src.coevolution import (
    CoevolutionSimulator,
    CoevolutionConfig,
    CoevolutionAnalyzer,
)
from src.evaluation import CoevolutionVisualizer, ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_attack(attack_config: dict) -> "BaseAttack":
    """Create attack instance from config."""
    attacker_knowledge = AttackerKnowledge(
        knows_aggregation=attack_config.get("knows_aggregation", True),
        knows_defense=attack_config.get("knows_defense", True),
        knows_thresholds=attack_config.get("knows_thresholds", False),
        knows_data_distribution=attack_config.get("knows_data_distribution", True),
    )

    attack_type = attack_config["type"]

    if attack_type == "label_flip":
        return DefenseAwareLabelFlip(
            attacker_knowledge=attacker_knowledge,
            attack_config=AttackConfig(attack_type="label_flip"),
            flip_ratio=attack_config.get("flip_ratio", 0.3),
            evasion_strategy=attack_config.get("evasion_strategy", "stay_under_bound"),
        )

    elif attack_type == "backdoor":
        return DefenseAwareBackdoor(
            attacker_knowledge=attacker_knowledge,
            attack_config=AttackConfig(attack_type="backdoor"),
            trigger_pattern=attack_config.get("trigger_pattern", "semantic"),
            injection_rate=attack_config.get("injection_rate", 0.1),
            gradual=attack_config.get("gradual", True),
        )

    elif attack_type == "model_poisoning":
        return DefenseAwareModelPoisoning(
            attacker_knowledge=attacker_knowledge,
            attack_config=AttackConfig(attack_type="model_poisoning"),
            poison_strength=attack_config.get("poison_strength", 5.0),
            norm_bound_aware=attack_config.get("norm_bound_aware", True),
        )

    elif attack_type == "evasion_poisoning":
        return EvasionPoisoningCombo(
            attacker_knowledge=attacker_knowledge,
            attack_config=AttackConfig(attack_type="evasion_poisoning"),
            evasion_method=attack_config.get("evasion_method", "pgd"),
        )

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def create_defense(defense_config: dict) -> "BaseDefense":
    """Create defense instance from config."""
    defender_observability = DefenderObservability(
        sees_gradients=defense_config.get("sees_gradients", True),
        sees_updates=defense_config.get("sees_updates", True),
        sees_client_ids=defense_config.get("sees_client_ids", True),
    )

    defense_type = defense_config["type"]

    if defense_type == "multi_round_anomaly":
        return MultiRoundAnomalyDetection(
            defender_observability=defender_observability,
            window_size=defense_config.get("window_size", 10),
            threshold_method=defense_config.get("threshold_method", "adaptive"),
        )

    elif defense_type == "honeypot":
        return HoneypotDefense(
            defender_observability=defender_observability,
            num_honeypots=defense_config.get("num_honeypots", 3),
            deviation_threshold=defense_config.get("deviation_threshold", 2.0),
        )

    elif defense_type == "gradient_forensics":
        return GradientForensics(
            defender_observability=defender_observability,
            analysis_method=defense_config.get("analysis_method", "pca"),
            coordination_threshold=defense_config.get("coordination_threshold", 0.9),
        )

    else:
        raise ValueError(f"Unknown defense type: {defense_type}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run co-evolution experiments")
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=["label_flip", "backdoor", "model_poisoning", "evasion_poisoning"],
        help="Attack type",
    )
    parser.add_argument(
        "--defense",
        type=str,
        required=True,
        choices=["multi_round_anomaly", "honeypot", "gradient_forensics"],
        help="Defense type",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of co-evolution rounds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate reports",
    )

    args = parser.parse_args()

    logger.info(f"Starting co-evolution experiment: {args.attack} vs {args.defense}")

    # Create attack and defense
    attack_config = {"type": args.attack}
    defense_config = {"type": args.defense}

    attack = create_attack(attack_config)
    defense = create_defense(defense_config)

    # Create simulator config
    sim_config = CoevolutionConfig(
        num_rounds=args.rounds,
        num_clients=10,
        num_malicious=2,
        num_samples=10000,
        num_features=100,
    )

    # Run simulation
    simulator = CoevolutionSimulator(sim_config, attack, defense)
    result = simulator.run()

    # Analyze results
    analyzer = CoevolutionAnalyzer(result)

    # Print summary
    print("\n")
    print("=" * 60)
    print(analyzer.generate_summary())
    print("=" * 60)

    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        visualizer = CoevolutionVisualizer(result.history, output_dir=args.output)
        visualizer.plot_metrics_over_rounds()
        visualizer.plot_costs_over_rounds()
        visualizer.plot_attack_defense_dynamics()
        visualizer.plot_equilibrium_analysis()

    # Generate reports
    if args.report:
        logger.info("Generating reports...")
        report_gen = ReportGenerator(analyzer, output_dir=args.output)
        report_gen.save_text_report()
        report_gen.save_markdown_report()
        report_gen.save_csv_results()
        report_gen.save_latex_table(table_type="summary")
        report_gen.save_latex_table(table_type="detailed")

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
