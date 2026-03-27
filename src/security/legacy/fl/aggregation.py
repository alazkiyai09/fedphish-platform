"""Aggregation strategies for federated learning."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AggregationStrategy:
    """Base class for aggregation strategies."""

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.

        Args:
            client_updates: List of (client_id, parameters) tuples

        Returns:
            Aggregated parameters
        """
        raise NotImplementedError


class FedAvg(AggregationStrategy):
    """Federated Averaging aggregation."""

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """Average all client updates equally."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        num_clients = len(client_updates)

        # Initialize aggregated parameters
        aggregated = {}
        for name, param in client_updates[0][1].items():
            aggregated[name] = torch.zeros_like(param)

        # Sum all parameters
        for client_id, params in client_updates:
            for name, param in params.items():
                aggregated[name] += param

        # Average
        for name in aggregated:
            aggregated[name] /= num_clients

        return aggregated


class Krum(AggregationStrategy):
    """
    Krum aggregation for Byzantine-robust FL.

    Selects the update closest to the most updates.
    """

    def __init__(self, num_malicious: int = 2):
        """
        Initialize Krum.

        Args:
            num_malicious: Estimated number of malicious clients
        """
        self.num_malicious = num_malicious

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Select the client update with minimum distance to others.

        Args:
            client_updates: List of (client_id, parameters) tuples

        Returns:
            Selected parameters
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        num_clients = len(client_updates)
        num_closest = num_clients - self.num_malicious - 2

        if num_closest <= 0:
            num_closest = num_clients // 2

        # Flatten parameters for distance computation
        flattened_updates = []
        for client_id, params in client_updates:
            flattened = torch.cat([param.flatten() for param in params.values()])
            flattened_updates.append(flattened)

        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores (sum of distances to closest num_closest updates)
        scores = []
        for i in range(num_clients):
            sorted_dist = np.sort(distances[i])
            score = np.sum(sorted_dist[:num_closest])
            scores.append(score)

        # Select update with minimum score
        selected_idx = np.argmin(scores)
        selected_params = client_updates[selected_idx][1]

        logger.debug(f"Krum selected client {client_updates[selected_idx][0]}")

        return selected_params


class MultiKrum(AggregationStrategy):
    """
    Multi-Krum aggregation.

    Averages multiple updates selected by Krum.
    """

    def __init__(self, num_malicious: int = 2, num_selected: int = 5):
        """
        Initialize Multi-Krum.

        Args:
            num_malicious: Estimated number of malicious clients
            num_selected: Number of updates to average
        """
        self.num_malicious = num_malicious
        self.num_selected = num_selected

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """Average top-k updates by Krum score."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        num_clients = len(client_updates)
        num_closest = num_clients - self.num_malicious - 2

        if num_closest <= 0:
            num_closest = num_clients // 2

        # Flatten parameters for distance computation
        flattened_updates = []
        for client_id, params in client_updates:
            flattened = torch.cat([param.flatten() for param in params.values()])
            flattened_updates.append(flattened)

        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores
        scores = []
        for i in range(num_clients):
            sorted_dist = np.sort(distances[i])
            score = np.sum(sorted_dist[:num_closest])
            scores.append(score)

        # Select top-k updates
        num_to_select = min(self.num_selected, num_clients)
        top_indices = np.argsort(scores)[:num_to_select]

        # Average selected updates
        aggregated = {}
        for name, param in client_updates[top_indices[0]][1].items():
            aggregated[name] = torch.zeros_like(param)

        for idx in top_indices:
            params = client_updates[idx][1]
            for name, param in params.items():
                aggregated[name] += param

        for name in aggregated:
            aggregated[name] /= num_to_select

        logger.debug(f"Multi-Krum selected {num_to_select} clients")

        return aggregated


class TrimmedMean(AggregationStrategy):
    """
    Trimmed Mean aggregation.

    Removes smallest and largest values, then averages.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize Trimmed Mean.

        Args:
            trim_ratio: Ratio of updates to trim from each end
        """
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """Trim and average client updates."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        num_clients = len(client_updates)
        num_to_trim = int(num_clients * self.trim_ratio)

        # Collect all parameter values
        param_dict: Dict[str, List[torch.Tensor]] = {}
        for client_id, params in client_updates:
            for name, param in params.items():
                if name not in param_dict:
                    param_dict[name] = []
                param_dict[name].append(param)

        # Trim and average
        aggregated = {}
        for name, param_list in param_dict.items():
            # Stack parameters
            stacked = torch.stack(param_list)

            # Compute mean for each element
            mean_param = torch.mean(stacked, dim=0)

            # Trim: remove values far from mean
            distances = torch.norm(stacked - mean_param, dim=tuple(range(1, stacked.ndim)))
            sorted_indices = torch.argsort(distances)

            # Remove top and bottom
            if num_to_trim > 0:
                keep_indices = sorted_indices[num_to_trim:-num_to_trim]
                if len(keep_indices) == 0:
                    keep_indices = sorted_indices[num_to_trim:]
                if len(keep_indices) == 0:
                    keep_indices = sorted_indices[:-num_to_trim]
                if len(keep_indices) == 0:
                    keep_indices = sorted_indices

                trimmed = stacked[keep_indices]
                aggregated[name] = torch.mean(trimmed, dim=0)
            else:
                aggregated[name] = mean_param

        logger.debug(f"Trimmed Mean trimmed {num_to_trim} updates")

        return aggregated
