"""Reproducibility utilities for the benchmark."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> int:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value

    Returns:
        The seed that was set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_seed(base_seed: int, run_id: int, experiment_id: Optional[int] = None) -> int:
    """
    Generate a deterministic seed for a specific experiment run.

    Args:
        base_seed: Base seed from config
        run_id: Run number (0 to num_runs-1)
        experiment_id: Optional experiment identifier

    Returns:
        Generated seed
    """
    seed = base_seed + run_id * 1000
    if experiment_id is not None:
        seed += experiment_id * 10000
    return seed
