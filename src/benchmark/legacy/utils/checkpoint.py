"""Checkpoint management for caching intermediate results."""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manage model checkpoints and cached results."""

    def __init__(self, cache_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            cache_dir: Directory to store checkpoints
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self,
        experiment_name: str,
        run_id: int,
        round_num: Optional[int] = None
    ) -> Path:
        """
        Get checkpoint path for an experiment.

        Args:
            experiment_name: Name of the experiment
            run_id: Run number
            round_num: Optional round number

        Returns:
            Checkpoint file path
        """
        round_suffix = f"_round{round_num}" if round_num is not None else ""
        filename = f"{experiment_name}_run{run_id}{round_suffix}.pkl"
        return self.cache_dir / filename

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        state_dict: Dict[str, Any]
    ) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint_path: Path to save checkpoint
            state_dict: State dictionary to save
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to load checkpoint from

        Returns:
            State dictionary if exists, None otherwise
        """
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def save_model(
        self,
        model: Any,
        model_name: str,
        run_id: int
    ) -> Path:
        """
        Save PyTorch model.

        Args:
            model: PyTorch model
            model_name: Name of the model
            run_id: Run number

        Returns:
            Path to saved model
        """
        model_dir = self.cache_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"run{run_id}.pt"
        torch.save(model.state_dict(), model_path)
        return model_path

    def load_model(
        self,
        model: Any,
        model_path: Path
    ) -> bool:
        """
        Load PyTorch model.

        Args:
            model: PyTorch model to load into
            model_path: Path to load from

        Returns:
            True if loaded successfully, False otherwise
        """
        if not model_path.exists():
            return False

        model.load_state_dict(torch.load(model_path, weights_only=True))
        return True

    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """
        Clear cache directory.

        Args:
            older_than_days: Only clear files older than this many days
        """
        if older_than_days is None:
            # Clear all
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        else:
            # Clear old files
            cutoff = datetime.now().timestamp() - (older_than_days * 24 * 3600)
            for file in self.cache_dir.glob("*"):
                if file.is_file() and file.stat().st_mtime < cutoff:
                    file.unlink()

    def cache_exists(
        self,
        experiment_name: str,
        run_id: int
    ) -> bool:
        """
        Check if cached result exists.

        Args:
            experiment_name: Name of the experiment
            run_id: Run number

        Returns:
            True if cache exists, False otherwise
        """
        checkpoint_path = self.get_checkpoint_path(experiment_name, run_id)
        return checkpoint_path.exists()
