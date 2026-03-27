"""Evaluation module update."""

from .metrics import compute_metrics
from .fairness import per_bank_metrics, compute_worst_case_accuracy, compute_fairness_gap, is_fair
from .privacy_tracker import PrivacyBudgetTracker, PrivacyBudget
from .statistical_tests import compute_mean_std, paired_t_test, anova_test, compute_confidence_interval

__all__ = [
    'compute_metrics',
    'per_bank_metrics',
    'compute_worst_case_accuracy',
    'compute_fairness_gap',
    'is_fair',
    'PrivacyBudgetTracker',
    'PrivacyBudget',
    'compute_mean_std',
    'paired_t_test',
    'anova_test',
    'compute_confidence_interval'
]
