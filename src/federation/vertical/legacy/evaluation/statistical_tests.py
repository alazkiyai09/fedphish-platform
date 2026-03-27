"""
Statistical testing for federated learning results.

Computes statistical significance across multiple runs.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation.

    Args:
        values: List of values

    Returns:
        Tuple of (mean, std)
    """
    return np.mean(values), np.std(values)


def paired_t_test(values_a: List[float],
                 values_b: List[float],
                 alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Perform paired t-test between two sets of results.

    Args:
        values_a: First set of values
        values_b: Second set of values
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    is_significant = p_value < alpha

    return t_stat, is_significant


def anova_test(groups: List[List[float]],
              alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Perform one-way ANOVA across multiple groups.

    Args:
        groups: List of value groups (e.g., different privacy settings)
        alpha: Significance level

    Returns:
        Tuple of (f_statistic, is_significant)
    """
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    is_significant = p_value < alpha

    return f_stat, is_significant


def compute_confidence_interval(values: List[float],
                              confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.

    Args:
        values: List of values
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(values)
    std_err = stats.sem(values)
    n = len(values)

    # t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * std_err

    return (mean - margin, mean + margin)


def is_significantly_different(values_a: List[float],
                             values_b: List[float],
                             alpha: float = 0.05) -> bool:
    """
    Test if two sets of values are significantly different.

    Args:
        values_a: First set of values
        values_b: Second set of values
        alpha: Significance level

    Returns:
        True if significantly different
    """
    _, is_significant = paired_t_test(values_a, values_b, alpha)
    return is_significant
