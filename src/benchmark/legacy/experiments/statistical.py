"""Statistical analysis utilities."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Statistical significance testing for benchmark results."""

    @staticmethod
    def compare_methods(
        results: pd.DataFrame,
        metric: str,
        method_a: str,
        method_b: str,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Compare two methods using statistical tests.

        Args:
            results: Results dataframe
            metric: Metric to compare
            method_a: First method name
            method_b: Second method name
            alpha: Significance level

        Returns:
            Test statistics and p-values
        """
        # Get results for each method
        results_a = results[results["model_type"] == method_a][f"{metric}_mean"].values
        results_b = results[results["model_type"] == method_b][f"{metric}_mean"].values

        # Perform paired t-test (assuming same data splits)
        t_stat, p_value = stats.ttest_rel(results_a, results_b)

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_p_value = stats.wilcoxon(results_a, results_b)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(results_a) ** 2 + np.std(results_b) ** 2) / 2)
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std

        return {
            "t_statistic": float(t_stat),
            "t_p_value": float(p_value),
            "significant": p_value < alpha,
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p_value": float(w_p_value),
            "cohens_d": float(cohens_d),
        }

    @staticmethod
    def compute_confidence_interval(
        values: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute confidence interval.

        Args:
            values: Sample values
            confidence: Confidence level

        Returns:
            CI bounds
        """
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)

        # Use t-distribution for small samples
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_critical * (std / np.sqrt(n))

        return {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(mean - margin_of_error),
            "ci_upper": float(mean + margin_of_error),
            "margin_of_error": float(margin_of_error),
        }

    @staticmethod
    def anova_test(
        results: pd.DataFrame,
        metric: str,
        group_column: str = "model_type"
    ) -> Dict[str, float]:
        """
        Perform ANOVA test across multiple groups.

        Args:
            results: Results dataframe
            metric: Metric to compare
            group_column: Column to group by

        Returns:
            ANOVA test results
        """
        groups = []
        for group_name, group_data in results.groupby(group_column):
            groups.append(group_data[f"{metric}_mean"].values)

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def compute_ranking(
        results: pd.DataFrame,
        metric: str,
        group_column: str = "model_type"
    ) -> pd.DataFrame:
        """
        Compute ranking of methods based on metric.

        Args:
            results: Results dataframe
            metric: Metric to rank by
            group_column: Column to group by

        Returns:
            Ranked dataframe
        """
        # Group by configuration and compute average metric
        grouped = results.groupby(group_column).agg({
            f"{metric}_mean": "mean",
            f"{metric}_std": "mean",
        })

        # Sort by metric (descending)
        ascending = metric in ["training_time", "communication_cost"]
        grouped = grouped.sort_values(f"{metric}_mean", ascending=ascending)

        # Add rank
        grouped["rank"] = range(1, len(grouped) + 1)

        return grouped.reset_index()
