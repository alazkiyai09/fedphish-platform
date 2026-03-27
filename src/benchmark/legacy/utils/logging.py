"""Logging utilities for MLflow integration."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logger


def log_experiment(
    config: DictConfig,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    artifacts: Optional[list] = None
) -> None:
    """
    Log experiment to MLflow.

    Args:
        config: Experiment configuration
        metrics: Dictionary of metrics to log
        params: Optional parameters to log
        artifacts: Optional list of artifact paths
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Start run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_type": config.model.type,
            "federation_type": config.federation.type,
            "data_distribution": config.dataset.partition.get("type", "iid"),
            "attack_type": str(config.attack.type if config.get("attack") else "none"),
            "num_clients": config.federation.client.num_clients,
            "num_rounds": config.federation.server.num_rounds,
        })

        # Log additional params if provided
        if params:
            mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts if provided
        if artifacts:
            for artifact in artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)

    logger.info(f"Logged experiment to MLflow: {metrics}")
