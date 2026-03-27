"""Random seed management."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed(base_seed: int, run_id: int, offset: Optional[int] = None) -> int:
    """
    Get seed for specific run.

    Args:
        base_seed: Base random seed
        run_id: Run identifier
        offset: Optional offset

    Returns:
        Derived seed
    """
    seed = base_seed + run_id * 1000
    if offset is not None:
        seed += offset
    return seed
