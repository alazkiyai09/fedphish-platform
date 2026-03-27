"""Report generation for co-evolution results."""

import logging
from pathlib import Path
from typing import Any, Dict

from ..coevolution.analyzer import CoevolutionAnalyzer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports from co-evolution results."""

    def __init__(
        self,
        analyzer: CoevolutionAnalyzer,
        output_dir: str = "./results",
    ):
        """
        Initialize report generator.

        Args:
            analyzer: Co-evolution analyzer
            output_dir: Output directory for reports
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(self) -> str:
        """
        Generate text summary report.

        Returns:
            Report text
        """
        return self.analyzer.generate_summary()

    def save_text_report(self, filename: str = "coevolution_report.txt") -> None:
        """
        Save text report to file.

        Args:
            filename: Output filename
        """
        report = self.generate_text_report()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Saved text report to {output_path}")

    def generate_latex_table(
        self,
        table_type: str = "summary",
    ) -> str:
        """
        Generate LaTeX table.

        Args:
            table_type: Type of table ("summary", "detailed")

        Returns:
            LaTeX table string
        """
        if table_type == "summary":
            return self._generate_summary_table()
        elif table_type == "detailed":
            return self._generate_detailed_table()
        else:
            raise ValueError(f"Unknown table type: {table_type}")

    def _generate_summary_table(self) -> str:
        """Generate summary LaTeX table."""
        final_metrics = self.analyzer.compute_final_metrics()
        trends = self.analyzer.analyze_trends()
        equilibrium = self.analyzer.compute_equilibrium_metrics()

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Co-evolution Summary}\n"
        latex += "\\label{tab:coevolution_summary}\n"
        latex += "\\begin{tabular}{ll}\n"
        latex += "\\toprule\n"
        latex += "Metric & Value \\\\\n"
        latex += "\\midrule\n"

        # Final metrics
        for key, value in final_metrics.items():
            if isinstance(value, float):
                latex += f"{key.replace('_', ' ').title()} & {value:.4f} \\\\\n"
            else:
                latex += f"{key.replace('_', ' ').title()} & {value} \\\\\n"

        latex += "\\midrule\n"

        # Equilibrium
        latex += f"Equilibrium Reached & {'Yes' if equilibrium.get('equilibrium_reached', False) else 'No'} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def _generate_detailed_table(self) -> str:
        """Generate detailed LaTeX table with per-round metrics."""
        history = self.analyzer.history

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Detailed Co-evolution Results}\n"
        latex += "\\label{tab:coevolution_detailed}\n"
        latex += "\\begin{tabular}{ccccccc}\n"
        latex += "\\toprule\n"
        latex += "Round & ASR & DR & Acc & Att. Cost & Def. Cost & Overhead \\\\\n"
        latex += "\\midrule\n"

        for round_metrics in history.rounds:
            latex += (
                f"{round_metrics.round_num} & "
                f"{round_metrics.attack_success_rate:.3f} & "
                f"{round_metrics.detection_rate:.3f} & "
                f"{round_metrics.model_accuracy:.3f} & "
                f"{round_metrics.attacker_cost:.2f} & "
                f"{round_metrics.defender_cost:.2f} & "
                f"{round_metrics.defense_overhead:.4f} \\\\\n"
            )

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def save_latex_table(
        self,
        table_type: str = "summary",
        filename: str = None,
    ) -> None:
        """
        Save LaTeX table to file.

        Args:
            table_type: Type of table
            filename: Output filename (default: coevolution_table.tex)
        """
        if filename is None:
            filename = f"coevolution_{table_type}_table.tex"

        latex = self.generate_latex_table(table_type)
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(latex)

        logger.info(f"Saved LaTeX table to {output_path}")

    def generate_markdown_report(self) -> str:
        """
        Generate Markdown report.

        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Co-evolution Simulation Report\n")

        final_metrics = self.analyzer.compute_final_metrics()
        trends = self.analyzer.analyze_trends()
        arms_race = self.analyzer.identify_arms_race()
        equilibrium = self.analyzer.compute_equilibrium_metrics()

        lines.append("## Final Metrics\n")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
            else:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")

        lines.append("\n## Trends\n")
        for key, value in trends.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")

        lines.append("\n## Arms Race Analysis\n")
        if arms_race.get("arms_race", False):
            lines.append("- **Arms Race Detected**: Yes\n")
            lines.append(f"- **Direction Changes**: {arms_race.get('total_direction_changes', 0)}\n")
        else:
            lines.append("- **Arms Race Detected**: No\n")

        lines.append("\n## Equilibrium\n")
        if equilibrium.get("equilibrium_reached", False):
            lines.append(f"- **Equilibrium Reached**: Yes (round {equilibrium.get('equilibrium_round', 'N/A')})\n")
        else:
            lines.append("- **Equilibrium Reached**: No\n")

        return "".join(lines)

    def save_markdown_report(self, filename: str = "coevolution_report.md") -> None:
        """
        Save Markdown report to file.

        Args:
            filename: Output filename
        """
        report = self.generate_markdown_report()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Saved Markdown report to {output_path}")

    def generate_csv_results(self) -> str:
        """
        Generate CSV results.

        Returns:
            CSV string
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Round",
            "Attack_Success_Rate",
            "Detection_Rate",
            "False_Positive_Rate",
            "Model_Accuracy",
            "Attacker_Cost",
            "Defender_Cost",
            "Defense_Overhead",
        ])

        # Data
        for round_metrics in self.analyzer.history.rounds:
            writer.writerow([
                round_metrics.round_num,
                round_metrics.attack_success_rate,
                round_metrics.detection_rate,
                round_metrics.false_positive_rate,
                round_metrics.model_accuracy,
                round_metrics.attacker_cost,
                round_metrics.defender_cost,
                round_metrics.defense_overhead,
            ])

        return output.getvalue()

    def save_csv_results(self, filename: str = "coevolution_results.csv") -> None:
        """
        Save CSV results to file.

        Args:
            filename: Output filename
        """
        csv = self.generate_csv_results()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(csv)

        logger.info(f"Saved CSV results to {output_path}")
