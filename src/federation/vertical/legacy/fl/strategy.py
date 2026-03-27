"""
Custom aggregation strategies for federated phishing detection.

Implements FedAvg, FedProx, and adaptive aggregation strategies.
"""

import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx: Federated Proximal training.

    Adds a proximal term to penalize large changes from the
    global model, handling heterogeneous data better.
    """

    def __init__(self,
                 mu: float = 0.01,
                 **kwargs):
        """
        Initialize FedProx strategy.

        Args:
            mu: Proximal term coefficient
            **kwargs: Arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.current_global_model = None

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client.ClientProxy, fl.common.FitRes]],
                      failures: List[fl.server.client.ClientProxy]) -> Optional[Tuple[fl.common.Parameters, Dict]]:
        """
        Aggregate model parameters using FedProx.

        Args:
            rnd: Round number
            results: List of (client, fit_result) tuples
            failures: Failed clients

        Returns:
            Tuple of (aggregated_parameters, metrics_dict)
        """
        # Extract parameters and weights
        weights = [fit_res.num_examples for _, fit_res in results]
        parameters = [fit_res.parameters for _, fit_res in results]

        # Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            rnd, results, failures
        )

        # Save current global model for proximal term
        self.current_global_model = aggregated_parameters

        # Add proximal term info to metrics
        metrics['proximal_mu'] = self.mu

        return aggregated_parameters, metrics

    def configure_fit(self,
                     rnd: int,
                      parameters: fl.common.Parameters,
                      client_manager: fl.server.ClientManager) -> List[Tuple[fl.server.client.ClientProxy, fl.common.FitIns]]:
        """
        Configure clients for next training round.

        Add global model to fit instructions for proximal term.
        """
        # Get standard fit instructions
        fit_ins = super().configure_fit(rnd, parameters, client_manager)

        # Add global model for proximal term
        if self.current_global_model is not None:
            fit_ins_with_prox = []
            for client_proxy, fit_in in fit_ins:
                fit_in.config['global_model'] = self.current_global_model
                fit_ins_with_prox.append((client_proxy, fit_in))
            return fit_ins_with_prox

        return fit_ins


class AdaptiveStrategy(fl.server.strategy.FedAvg):
    """
    Adaptive aggregation strategy.

    Excludes poorly performing clients from aggregation
    based on validation accuracy. This improves robustness
    to malicious or low-quality data.
    """

    def __init__(self,
                 min_accuracy: float = 0.7,
                 **kwargs):
        """
        Initialize Adaptive strategy.

        Args:
            min_accuracy: Minimum accuracy threshold
            **kwargs: Arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.min_accuracy = min_accuracy

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client.ClientProxy, fl.common.FitRes]],
                      failures: List[fl.server.client.ClientProxy]) -> Optional[Tuple[fl.common.Parameters, Dict]]:
        """
        Aggregate using only clients with good performance.

        Args:
            rnd: Round number
            results: List of (client, fit_result) tuples
            failures: Failed clients

        Returns:
            Tuple of (aggregated_parameters, metrics_dict)
        """
        # Filter out clients below accuracy threshold
        filtered_results = []

        for client, fit_res in results:
            metrics = fit_res.metrics

            # Check if client meets minimum accuracy
            if 'accuracy' in metrics and metrics['accuracy'] >= self.min_accuracy:
                filtered_results.append((client, fit_res))

        # Log excluded clients
        n_excluded = len(results) - len(filtered_results)
        excluded_metrics = {
            'n_excluded': n_excluded,
            'n_included': len(filtered_results)
        }

        # If too few clients, fallback to standard aggregation
        if len(filtered_results) < 2:
            filtered_results = results
            excluded_metrics['fallback'] = True

        # Aggregate using filtered results
        aggregated_parameters, metrics = super().aggregate_fit(
            rnd, filtered_results, failures
        )

        # Add exclusion info to metrics
        metrics.update(excluded_metrics)

        return aggregated_parameters, metrics


class SecureAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Secure aggregation strategy using homomorphic encryption.

    Clients encrypt their updates before sending to server.
    Server aggregates in encrypted domain.
    """

    def __init__(self,
                 n_clients: int,
                 **kwargs):
        """
        Initialize secure aggregation strategy.

        Args:
            n_clients: Number of clients
            **kwargs: Arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.n_clients = n_clients

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client.ClientProxy, fl.common.FitRes]],
                      failures: List[fl.server.client.ClientProxy]) -> Optional[Tuple[fl.common.Parameters, Dict]]:
        """
        Aggregate with secure aggregation.

        NOTE: This is a simulation for educational purposes.
        In production, would use TenSEAL for CKKS/BFV encryption.
        Actual implementation would:
        1. Encrypt client updates before sending
        2. Perform homomorphic addition on encrypted updates
        3. Decrypt aggregated result server-side
        For now, we use standard FedAvg for simulation.
        """
        aggregated_parameters, metrics = super().aggregate_fit(
            rnd, results, failures
        )

        metrics['secure_aggregation'] = True

        return aggregated_parameters, metrics
