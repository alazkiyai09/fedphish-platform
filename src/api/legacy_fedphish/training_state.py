"""
Training state tracking for FedPhish API.

Provides in-memory tracking of federated training status.
In production, this should be replaced with a persistent state store (Redis, database).
"""
from typing import Dict, Any
from threading import Lock
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current federated training state."""

    status: str = "idle"  # idle, running, completed, failed
    current_round: int = 0
    total_rounds: int = 20
    accuracy: float = 0.0
    loss: float = 0.0
    num_clients: int = 0
    start_time: str = ""
    error_message: str = ""


# Global state instance
_state: Dict[str, Any] = {
    "status": "idle",
    "current_round": 0,
    "total_rounds": 20,
    "accuracy": 0.0,
    "loss": 0.0,
    "num_clients": 0,
    "start_time": "",
    "error_message": "",
}
_state_lock = Lock()


def get_training_state() -> Dict[str, Any]:
    """
    Get current training state (thread-safe).

    Returns:
        Dictionary with current training status
    """
    with _state_lock:
        return _state.copy()


def update_training_state(
    status: str = None,
    current_round: int = None,
    total_rounds: int = None,
    accuracy: float = None,
    loss: float = None,
    num_clients: int = None,
    error_message: str = None,
) -> None:
    """
    Update training state (thread-safe).

    Args:
        status: Training status (idle, running, completed, failed)
        current_round: Current training round
        total_rounds: Total number of rounds
        accuracy: Current accuracy
        loss: Current loss
        num_clients: Number of active clients
        error_message: Error message if status is failed
    """
    with _state_lock:
        if status is not None:
            _state["status"] = status
            logger.info(f"Training status updated: {status}")

        if current_round is not None:
            _state["current_round"] = current_round

        if total_rounds is not None:
            _state["total_rounds"] = total_rounds

        if accuracy is not None:
            _state["accuracy"] = accuracy

        if loss is not None:
            _state["loss"] = loss

        if num_clients is not None:
            _state["num_clients"] = num_clients

        if error_message is not None:
            _state["error_message"] = error_message


def reset_training_state() -> None:
    """Reset training state to initial values."""
    with _state_lock:
        _state.update({
            "status": "idle",
            "current_round": 0,
            "total_rounds": 20,
            "accuracy": 0.0,
            "loss": 0.0,
            "num_clients": 0,
            "start_time": "",
            "error_message": "",
        })
    logger.info("Training state reset")


def start_training(total_rounds: int = 20, num_clients: int = 0) -> None:
    """
    Mark training as started.

    Args:
        total_rounds: Total number of training rounds
        num_clients: Number of participating clients
    """
    from datetime import datetime

    with _state_lock:
        _state["status"] = "running"
        _state["current_round"] = 0
        _state["total_rounds"] = total_rounds
        _state["num_clients"] = num_clients
        _state["start_time"] = datetime.utcnow().isoformat()
        _state["error_message"] = ""
    logger.info(f"Training started: {total_rounds} rounds, {num_clients} clients")


def complete_training(final_accuracy: float, final_loss: float) -> None:
    """
    Mark training as completed.

    Args:
        final_accuracy: Final accuracy achieved
        final_loss: Final loss achieved
    """
    with _state_lock:
        _state["status"] = "completed"
        _state["accuracy"] = final_accuracy
        _state["loss"] = final_loss
    logger.info(f"Training completed: accuracy={final_accuracy:.4f}, loss={final_loss:.4f}")


def fail_training(error_message: str) -> None:
    """
    Mark training as failed.

    Args:
        error_message: Error message describing the failure
    """
    with _state_lock:
        _state["status"] = "failed"
        _state["error_message"] = error_message
    logger.error(f"Training failed: {error_message}")
