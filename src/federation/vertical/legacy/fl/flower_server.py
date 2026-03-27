"""
Flower server implementation for federated phishing detection.

Coordinates training across 5 banks with custom aggregation strategies.
"""

import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Compute weighted average of metrics across clients.

    Args:
        metrics: List of (n_samples, metrics_dict) tuples

    Returns:
        Weighted average metrics
    """
    # Calculate total number of samples
    total_samples = sum(num_samples for num_samples, _ in metrics)

    # Compute weighted averages
    aggregated_metrics = {}

    for metric_name in metrics[0][1].keys():
        if metric_name == 'n_samples':
            continue

        weighted_sum = sum(
            m[metric_name] * num_samples
            for num_samples, m in metrics
        )
        aggregated_metrics[metric_name] = weighted_sum / total_samples

    aggregated_metrics['n_samples'] = total_samples

    return aggregated_metrics


def create_server(strategy: str = 'fedavg',
                  min_fit_clients: int = 3,
                  min_evaluate_clients: int = 2,
                  min_available_clients: int = 3,
                  n_rounds: int = 50,
                  fraction_fit: float = 1.0,
                  fraction_evaluate: float = 0.5) -> fl.server.Server:
    """
    Create Flower server with specified strategy.

    Args:
        strategy: 'fedavg', 'fedprox', or 'adaptive'
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum clients available
        n_rounds: Number of FL rounds
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation

    Returns:
        Configured Flower server
    """
    # Create strategy
    if strategy == 'fedavg':
        strategy_instance = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_fn=weighted_average,
            on_fit_config_fn=lambda rnd: {'rnd': rnd},
            on_evaluate_config_fn=lambda rnd: {'rnd': rnd}
        )
    elif strategy == 'fedprox':
        # Import FedProx strategy
        from .strategy import FedProxStrategy
        strategy_instance = FedProxStrategy(
            mu=0.01,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_fn=weighted_average
        )
    elif strategy == 'adaptive':
        from .strategy import AdaptiveStrategy
        strategy_instance = AdaptiveStrategy(
            min_accuracy=0.7,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_fn=weighted_average
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create server
    server = fl.server.Server(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy_instance,
        num_clients=min_available_clients
    )

    return server


def run_federated_simulation(num_rounds: int = 50,
                             n_clients: int = 5,
                             privacy_mechanism: str = 'none',
                             strategy: str = 'fedavg') -> Dict:
    """
    Run federated learning simulation.

    Args:
        num_rounds: Number of training rounds
        n_clients: Number of bank clients
        privacy_mechanism: Privacy mechanism to use
        strategy: Aggregation strategy

    Returns:
        Training results dictionary
    """
    # Import here to avoid circular imports
    from ..banks import GlobalBank, RegionalBank, DigitalBank, CreditUnion, InvestmentBank
    from .client_manager import ClientManager

    # Create clients
    client_manager = ClientManager(n_clients=n_clients)
    client_manager.create_all_clients(data_path='data/bank_datasets')

    # Get client functions
    client_fn = client_manager.get_client_fn(
        privacy_mechanism=privacy_mechanism
    )

    # Create server
    server = create_server(
        strategy=strategy,
        n_rounds=num_rounds
    )

    # Run simulation
    from flwr.simulation import start_simulation

    history = start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=server,
        client_resources={'num_cpus': 2, 'num_gpus': 0.5}
    )

    return {
        'history': history,
        'final_loss': history.losses_centralized[-1] if history.losses_centralized else None,
        'final_accuracy': history.metrics_centralized['accuracy'][-1] if history.metrics_centralized else None
    }
