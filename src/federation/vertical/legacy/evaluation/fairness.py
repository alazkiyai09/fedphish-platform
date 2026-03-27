"""
Fairness evaluation for federated phishing detection.

Computes per-bank metrics to ensure fairness across all banks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .metrics import compute_metrics


def per_bank_metrics(banks: List,
                     global_model,
                     test_loaders: List) -> pd.DataFrame:
    """
    Compute per-bank accuracy for fairness analysis.

    Args:
        banks: List of bank instances
        global_model: Trained global model
        test_loaders: List of test DataLoaders

    Returns:
        DataFrame with per-bank metrics
    """
    results = []

    for bank, test_loader in zip(banks, test_loaders):
        # Evaluate on this bank's test set
        import torch
        all_preds = []
        all_labels = []

        global_model.eval()
        device = next(global_model.parameters()).device
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(global_model.device)
                attention_mask = batch['attention_mask'].to(global_model.device)
                labels = batch['labels']

                logits = global_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds)
        )

        results.append({
            'bank': bank.profile.name,
            'bank_type': bank.profile.bank_type,
            'n_samples': len(test_loader.dataset),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'data_quality': bank.get_data_quality()
        })

    return pd.DataFrame(results)


def compute_worst_case_accuracy(per_bank_df: pd.DataFrame) -> float:
    """
    Compute worst-case accuracy across all banks.

    This is a key fairness metric - the worst-performing bank
    determines the overall system fairness.

    Args:
        per_bank_df: DataFrame from per_bank_metrics()

    Returns:
        Minimum accuracy across all banks
    """
    return per_bank_df['accuracy'].min()


def compute_fairness_gap(per_bank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute fairness metrics across banks.

    Args:
        per_bank_df: DataFrame from per_bank_metrics()

    Returns:
        Dictionary with fairness metrics
    """
    accuracies = per_bank_df['accuracy'].values

    return {
        'min_accuracy': accuracies.min(),
        'max_accuracy': accuracies.max(),
        'mean_accuracy': accuracies.mean(),
        'std_accuracy': accuracies.std(),
        'accuracy_gap': accuracies.max() - accuracies.min(),
        'coefficient_of_variation': accuracies.std() / accuracies.mean() if accuracies.mean() > 0 else 0
    }


def is_fair(per_bank_df: pd.DataFrame,
           max_accuracy_gap: float = 0.10) -> bool:
    """
    Determine if the model is fair across all banks.

    Fairness criterion: accuracy gap between best and worst bank
    should be less than max_accuracy_gap.

    Args:
        per_bank_df: DataFrame from per_bank_metrics()
        max_accuracy_gap: Maximum allowed accuracy gap

    Returns:
        True if model is fair, False otherwise
    """
    fairness_metrics = compute_fairness_gap(per_bank_df)

    return fairness_metrics['accuracy_gap'] <= max_accuracy_gap
