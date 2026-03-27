"""Single experiment runner."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..data import PhishingDataLoader, DataPartitioner
from ..models import (
    XGBoostModel,
    RandomForestModel,
    LogisticRegressionModel,
    TransformerModel,
    MultiAgentModel,
    PrivacyGBDTModel,
)
from ..fl import FedPhishClient, get_strategy
from ..attacks import LabelFlipAttack, BackdoorAttack, ModelPoisoningAttack
from ..metrics import compute_all_classification_metrics
from ..utils import set_seed, get_seed, CheckpointManager

logger = logging.getLogger(__name__)


class ExperimentResult:
    """Container for experiment results."""

    def __init__(self):
        self.metrics = {}
        self.config = {}
        self.training_time = 0.0
        self.timestamp = datetime.now().isoformat()


class ExperimentRunner:
    """Run a single experiment."""

    def __init__(self, config: DictConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.benchmark.cache_dir)

    def run(
        self,
        model_type: str,
        federation_type: str,
        data_distribution: str,
        attack_type: str = "none",
        privacy_mechanism: str = "none",
        run_id: int = 0,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            model_type: Type of model
            federation_type: Type of federation
            data_distribution: Data distribution
            attack_type: Attack type
            privacy_mechanism: Privacy mechanism
            run_id: Run number

        Returns:
            Experiment results
        """
        # Set seed for reproducibility
        seed = get_seed(self.config.benchmark.random_seed, run_id)
        set_seed(seed)

        result = ExperimentResult()
        result.config = {
            "model_type": model_type,
            "federation_type": federation_type,
            "data_distribution": data_distribution,
            "attack_type": attack_type,
            "privacy_mechanism": privacy_mechanism,
            "run_id": run_id,
            "seed": seed,
        }

        start_time = time.time()

        # Load data
        logger.info("Loading data...")
        data_loader = PhishingDataLoader(self.config)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()

        # Partition data
        partitioner = DataPartitioner(
            self.config.dataset.partition.num_clients,
            seed=seed
        )
        train_partitions = partitioner.partition(
            X_train, y_train, data_distribution
        )
        val_partitions = partitioner.partition(
            X_val, y_val, data_distribution
        )

        # Create model
        model = self._create_model(model_type)

        # Train based on federation type
        if federation_type == "centralized":
            metrics = self._train_centralized(model, X_train, y_train, X_val, y_val)
        elif federation_type == "local":
            metrics = self._train_local(
                model, X_train, y_train, X_val, y_val,
                train_partitions, val_partitions
            )
        else:
            metrics = self._train_federated(
                model,
                X_train, y_train,
                X_val, y_val,
                train_partitions,
                val_partitions,
                federation_type,
                attack_type,
                privacy_mechanism,
            )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self._evaluate_model(model, X_test, y_test)
        metrics.update(test_metrics)

        # Add training time
        result.training_time = time.time() - start_time
        metrics["training_time"] = result.training_time

        result.metrics = metrics

        # Log results
        logger.info(f"Experiment completed: {metrics}")

        return result

    def _create_model(self, model_type: str):
        """Create model instance."""
        if model_type == "xgboost":
            return XGBoostModel(self.config)
        elif model_type == "random_forest":
            return RandomForestModel(self.config)
        elif model_type == "logistic_regression":
            return LogisticRegressionModel(self.config)
        elif model_type == "transformer":
            return TransformerModel(self.config)
        elif model_type == "multi_agent":
            return MultiAgentModel(self.config)
        elif model_type == "privacy_gbdt":
            return PrivacyGBDTModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_centralized(
        self, model, X_train, y_train, X_val, y_val
    ) -> Dict[str, float]:
        """Train centralized model."""
        logger.info("Training centralized model...")
        model.fit(X_train, y_train, X_val, y_val)
        return model.evaluate(X_val, y_val)

    def _train_local(
        self, model, X_train, y_train, X_val, y_val,
        train_partitions, val_partitions
    ) -> Dict[str, float]:
        """Train local models (per bank)."""
        logger.info("Training local models...")

        client_metrics = []
        for client_id in train_partitions.keys():
            # Get client data using partition indices
            train_indices = train_partitions[client_id]
            X_train_client = X_train[train_indices]
            y_train_client = y_train[train_indices]

            val_indices = val_partitions[client_id]
            X_val_client = X_val[val_indices]
            y_val_client = y_val[val_indices]

            # Train client model
            client_model = self._create_model(self.config.model.type)
            client_model.fit(X_train_client, y_train_client, X_val_client, y_val_client)

            # Evaluate
            metrics = client_model.evaluate(X_val_client, y_val_client)
            client_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in client_metrics[0].keys():
            values = [m[key] for m in client_metrics]
            avg_metrics[key] = np.mean(values)

        return avg_metrics

    def _train_federated(
        self, model, X_train, y_train, X_val, y_val,
        train_partitions, val_partitions,
        federation_type, attack_type, privacy_mechanism
    ) -> Dict[str, float]:
        """Train federated model."""
        logger.info(f"Training federated model ({federation_type})...")

        # Note: This is a simplified simulation
        # In practice, you'd use Flower's actual client-server architecture

        num_clients = len(train_partitions)
        num_rounds = self.config.federation.server.num_rounds

        # Simulate FL training
        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")

            # Select clients (simplified - use all clients)
            client_updates = []

            for client_id in range(num_clients):
                # Get client data using partition indices
                client_indices = train_partitions[client_id]
                X_train_client = X_train[client_indices]
                y_train_client = y_train[client_indices]

                # Train locally
                model.fit(X_train_client, y_train_client)

                # Get parameters
                params = model.get_params()

                # Apply attack if needed
                if attack_type != "none":
                    params = self._apply_attack(attack_type, params)

                client_updates.append(params)

            # Aggregate (simplified FedAvg)
            aggregated_params = self._aggregate_updates(client_updates)
            model.set_params(aggregated_params)

        # Evaluate
        # Get first client's validation data
        client_id = 0
        client_indices = val_partitions[client_id]
        X_val_client = X_val[client_indices]
        y_val_client = y_val[client_indices]

        return model.evaluate(X_val_client, y_val_client)

    def _apply_attack(self, attack_type: str, params):
        """Apply attack to parameters."""
        if attack_type == "model_poisoning":
            # Scale parameters
            from ..data.attack_injector import scale_gradients
            return scale_gradients(params, -5.0)
        return params

    def _aggregate_updates(self, updates: list) -> list:
        """Aggregate client updates (simplified FedAvg)."""
        num_updates = len(updates)
        if num_updates == 0:
            return []

        # Check if updates are dictionaries (sklearn models) or lists (PyTorch models)
        if isinstance(updates[0], dict):
            # For sklearn models, return first update (simplified)
            # True model averaging for sklearn is complex and not standard
            return updates[0]
        else:
            # For PyTorch models, average parameters
            aggregated = []
            for i in range(len(updates[0])):
                params = [u[i] for u in updates if len(u) > i]
                if params:
                    avg_param = np.mean(params, axis=0)
                    aggregated.append(avg_param)
            return aggregated

    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate model on test set."""
        return model.evaluate(X_test, y_test)


def run_single_experiment(
    config: DictConfig,
    model_type: str,
    federation_type: str,
    data_distribution: str,
    attack_type: str = "none",
    privacy_mechanism: str = "none",
    run_id: int = 0,
) -> ExperimentResult:
    """
    Convenience function to run a single experiment.

    Args:
        config: Configuration
        model_type: Model type
        federation_type: Federation type
        data_distribution: Data distribution
        attack_type: Attack type
        privacy_mechanism: Privacy mechanism
        run_id: Run ID

    Returns:
        Experiment result
    """
    runner = ExperimentRunner(config)
    return runner.run(
        model_type=model_type,
        federation_type=federation_type,
        data_distribution=data_distribution,
        attack_type=attack_type,
        privacy_mechanism=privacy_mechanism,
        run_id=run_id,
    )
