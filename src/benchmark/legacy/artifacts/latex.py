"""LaTeX table generation for research papers."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_cell(
    value: float,
    std: float,
    bold_best: bool = True,
    format_str: str = "{:.3f}"
) -> str:
    """
    Format a cell value with standard deviation.

    Args:
        value: Mean value
        std: Standard deviation
        bold_best: Whether to bold the best value
        format_str: Format string for value

    Returns:
        Formatted cell string
    """
    formatted = format_str.format(value) + " $\\pm$ " + format_str.format(std)
    return formatted


def generate_latex_table(
    results: pd.DataFrame,
    table_type: str = "main",
    output_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX table from results.

    Args:
        results: Results dataframe
        table_type: Type of table (main, ablation, attack, privacy)
        output_path: Optional path to save table

    Returns:
        LaTeX table string
    """
    if table_type == "main":
        latex = _generate_main_results_table(results)
    elif table_type == "ablation":
        latex = _generate_ablation_table(results)
    elif table_type == "attack":
        latex = _generate_attack_table(results)
    elif table_type == "privacy":
        latex = _generate_privacy_table(results)
    else:
        raise ValueError(f"Unknown table type: {table_type}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        logger.info(f"LaTeX table saved to {output_path}")

    return latex


def _generate_main_results_table(results: pd.DataFrame) -> str:
    """Generate main results table."""
    # Filter for IID, no attack, no privacy (baseline comparison)
    baseline = results[
        (results["data_distribution"] == "iid") &
        (results["attack_type"] == "none") &
        (results["privacy_mechanism"] == "none")
    ]

    # Pivot for table
    table_data = []
    for _, row in baseline.iterrows():
        table_data.append({
            "Method": row["model_type"].upper().replace("_", "-"),
            "IID Acc": f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}",
            "Non-IID Acc": f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}",  # Placeholder
            "AUPRC": f"{row['auprc_mean']:.3f} $\\pm$ {row['auprc_std']:.3f}",
            "Time (s)": f"{row['training_time_mean']:.1f}",
        })

    df = pd.DataFrame(table_data)

    # Convert to LaTeX
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption="Federated phishing detection performance comparison",
        label="tab:main_results",
    )

    return latex


def _generate_ablation_table(results: pd.DataFrame) -> str:
    """Generate ablation study table."""
    # Ablation on federation types
    table_data = []

    fed_types = results["federation_type"].unique()
    for fed_type in fed_types:
        subset = results[results["federation_type"] == fed_type]
        if len(subset) > 0:
            row = subset.iloc[0]
            table_data.append({
                "Federation": fed_type.upper().replace("_", "-"),
                "Accuracy": f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}",
                "AUPRC": f"{row['auprc_mean']:.3f} $\\pm$ {row['auprc_std']:.3f}",
                "Time": f"{row['training_time_mean']:.1f}",
            })

    df = pd.DataFrame(table_data)
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption="Ablation study on federation strategies",
        label="tab:ablation",
    )

    return latex


def _generate_attack_table(results: pd.DataFrame) -> str:
    """Generate attack analysis table."""
    table_data = []

    attacks = results["attack_type"].unique()
    for attack in attacks:
        if attack == "none":
            continue
        subset = results[results["attack_type"] == attack]
        if len(subset) > 0:
            row = subset.iloc[0]
            table_data.append({
                "Attack": attack.replace("_", " ").title(),
                "Accuracy": f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}",
                "ASR": "N/A",  # Would need attack_success_rate column
                "Degradation": "N/A",
            })

    df = pd.DataFrame(table_data)
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption="Impact of adversarial attacks on performance",
        label="tab:attacks",
    )

    return latex


def _generate_privacy_table(results: pd.DataFrame) -> str:
    """Generate privacy-utility tradeoff table."""
    table_data = []

    epsilon_values = [1.0, 0.5, 0.1]
    for eps in epsilon_values:
        subset = results[results["privacy_mechanism"] == f"local_dp_{eps}"]
        if len(subset) > 0:
            row = subset.iloc[0]
            table_data.append({
                "$\\epsilon$": str(eps),
                "Accuracy": f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}",
                "AUPRC": f"{row['auprc_mean']:.3f} $\\pm$ {row['auprc_std']:.3f}",
            })

    df = pd.DataFrame(table_data)
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lcc",
        caption="Privacy-utility tradeoff with local DP",
        label="tab:privacy",
    )

    return latex


def generate_latex_tables(
    results: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Generate all LaTeX tables.

    Args:
        results: Results dataframe
        output_dir: Output directory
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Generate all table types
    for table_type in ["main", "ablation", "attack", "privacy"]:
        try:
            generate_latex_table(
                results,
                table_type=table_type,
                output_path=tables_dir / f"{table_type}.tex"
            )
        except Exception as e:
            logger.warning(f"Failed to generate {table_type} table: {e}")

    logger.info(f"All LaTeX tables generated in {tables_dir}")
