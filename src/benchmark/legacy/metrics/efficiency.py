"""Efficiency metrics: training time, communication cost."""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_training_time(
    start_time: float,
    end_time: float
) -> Dict[str, float]:
    """
    Compute training time metrics.

    Args:
        start_time: Training start time
        end_time: Training end time

    Returns:
        Time metrics
    """
    total_time = end_time - start_time

    return {
        "total_time": float(total_time),
        "total_time_minutes": float(total_time / 60),
        "total_time_hours": float(total_time / 3600),
    }


def compute_communication_cost(
    num_rounds: int,
    num_clients_per_round: int,
    parameter_sizes: List[int],
    message_size_bytes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute communication cost.

    Args:
        num_rounds: Number of communication rounds
        num_clients_per_round: Clients participating per round
        parameter_sizes: Sizes of parameters (in bytes)
        message_size_bytes: Size of each message (optional)

    Returns:
        Communication metrics
    """
    if message_size_bytes is None:
        message_size_bytes = sum(parameter_sizes)

    # Total messages: upload + download per round
    messages_per_round = num_clients_per_round * 2  # Upload + download
    total_messages = num_rounds * messages_per_round

    # Total bytes transferred
    total_bytes = total_messages * message_size_bytes

    return {
        "total_messages": int(total_messages),
        "total_bytes": float(total_bytes),
        "total_mb": float(total_bytes / (1024 * 1024)),
        "total_gb": float(total_bytes / (1024 * 1024 * 1024)),
        "avg_bytes_per_round": float(total_bytes / num_rounds),
    }


def compute_computation_cost(
    num_clients: int,
    local_epochs: int,
    batch_size: int,
    num_samples: int,
    avg_epoch_time: float
) -> Dict[str, float]:
    """
    Compute computation cost.

    Args:
        num_clients: Number of clients
        local_epochs: Local training epochs
        batch_size: Batch size
        num_samples: Number of samples per client
        avg_epoch_time: Average time per epoch (seconds)

    Returns:
        Computation metrics
    """
    # Batches per epoch
    batches_per_epoch = num_samples / batch_size

    # Total computation time across all clients
    total_computation_time = num_clients * local_epochs * avg_epoch_time

    return {
        "total_computation_time": float(total_computation_time),
        "total_computation_time_hours": float(total_computation_time / 3600),
        "batches_per_epoch": float(batches_per_epoch),
        "total_batches": float(num_clients * local_epochs * batches_per_epoch),
    }


class Timer:
    """Timer for measuring execution time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        """
        Get elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
