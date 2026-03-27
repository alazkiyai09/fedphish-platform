"""Evaluation metrics: classification, privacy, robustness, efficiency, fairness."""

from .classification import (
    compute_accuracy,
    compute_auprc,
    compute_recall_at_fpr,
    compute_all_classification_metrics,
)
from .privacy import (
    compute_epsilon,
    compute_privacy_loss,
    compute_information_leakage,
)
from .robustness import (
    compute_attack_success_rate,
    compute_accuracy_degradation,
)
from .efficiency import (
    compute_training_time,
    compute_communication_cost,
    compute_computation_cost,
)
from .fairness import (
    compute_per_bank_accuracy,
    compute_accuracy_variance,
)

__all__ = [
    "compute_accuracy",
    "compute_auprc",
    "compute_recall_at_fpr",
    "compute_all_classification_metrics",
    "compute_epsilon",
    "compute_privacy_loss",
    "compute_information_leakage",
    "compute_attack_success_rate",
    "compute_accuracy_degradation",
    "compute_training_time",
    "compute_communication_cost",
    "compute_computation_cost",
    "compute_per_bank_accuracy",
    "compute_accuracy_variance",
]
