"""
Malicious client implementation for robustness testing.

From Day 11: Poisoning attacks and defenses.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class MaliciousClient:
    """
    Malicious client that attacks federated learning.

    Attack types:
    - Label flipping: Send inverted gradients
    - Backdoor: Insert trigger pattern
    - Byzantine: Send random/zero updates
    """

    def __init__(self,
                 attack_type: str = 'label_flip',
                 poisoning_rate: float = 0.1):
        """
        Initialize malicious client.

        Args:
            attack_type: 'label_flip', 'backdoor', 'byzantine'
            poisoning_rate: Fraction of samples to poison
        """
        self.attack_type = attack_type
        self.poisoning_rate = poisoning_rate

    def poison_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Poison gradients before sending to server.

        Args:
            gradients: Original gradients

        Returns:
            Poisoned gradients
        """
        poisoned = []

        for grad in gradients:
            if self.attack_type == 'label_flip':
                # Invert gradients
                poisoned.append(-grad * 0.5)
            elif self.attack_type == 'byzantine':
                # Send random gradients
                noise = torch.randn_like(grad) * 0.1
                poisoned.append(grad + noise)
            elif self.attack_type == 'backdoor':
                # Add backdoor trigger pattern
                backdoor_pattern = torch.ones_like(grad) * 0.01
                poisoned.append(grad + backdoor_pattern)
            else:
                poisoned.append(grad)

        return poisoned

    def poison_updates(self, updates: List[np.ndarray]) -> List[np.ndarray]:
        """
        Poison model updates before sending.

        Args:
            updates: Original model updates

        Returns:
            Poisoned updates
        """
        poisoned = []

        for update in updates:
            if self.attack_type == 'byzantine':
                # Send zeros or random values
                if np.random.rand() < 0.5:
                    poisoned.append(np.zeros_like(update))
                else:
                    noise = np.random.randn(*update.shape) * 0.1
                    poisoned.append(update + noise)
            else:
                poisoned.append(update)

        return poisoned


class DefenseMechanism:
    """
    Defense mechanisms against malicious clients.

    From Day 11: Krum, Multi-Krum, MFC, etc.
    """

    def __init__(self, n_clients: int, n_malicious: int = 1):
        """
        Initialize defense.

        Args:
            n_clients: Total number of clients
            n_malicious: Number of malicious clients
        """
        self.n_clients = n_clients
        self.n_malicious = n_malicious

    def krum(self,
             updates: List[np.ndarray],
             n_malicious: int = 1) -> int:
        """
        Krum defense: Select update closest to others.

        Args:
            updates: List of client updates
            n_malicious: Number of malicious clients tolerated

        Returns:
            Index of best client (most similar to others)
        """
        # Compute pairwise distances
        n_updates = len(updates)
        distances = np.zeros((n_updates, n_updates))

        for i in range(n_updates):
            for j in range(n_updates):
                if i != j:
                    # Euclidean distance
                    distances[i, j] = np.linalg.norm(updates[i] - updates[j])

        # For each update, compute sum of distances to closest n-malicious-1 others
        scores = np.zeros(n_updates)

        for i in range(n_updates):
            # Sort distances for client i
            sorted_indices = np.argsort(distances[i])
            # Take closest n-malicious clients
            closest_indices = sorted_indices[:n_updates - n_malicious]
            scores[i] = np.sum(distances[i, closest_indices])

        # Return client with minimum score
        return np.argmin(scores)

    def multi_krum(self,
                  updates: List[np.ndarray],
                  n_malicious: int = 1) -> List[int]:
        """
        Multi-Krum: Select multiple best updates.

        Args:
            updates: List of client updates
            n_malicious: Number of malicious clients

        Returns:
            List of indices of best clients
        """
        # Compute distances
        n_updates = len(updates)
        distances = np.zeros((n_updates, n_updates))

        for i in range(n_updates):
            for j in range(n_updates):
                if i != j:
                    distances[i, j] = np.linalg.norm(updates[i] - updates[j])

        # Score each client
        scores = np.zeros(n_updates)
        for i in range(n_updates):
            sorted_indices = np.argsort(distances[i])
            closest_indices = sorted_indices[:n_updates - n_malicious]
            scores[i] = np.sum(distances[i, closest_indices])

        # Select top n_updates - n_malicious clients
        n_selected = n_updates - n_malicious
        best_indices = np.argsort(scores)[:n_selected]

        return best_indices.tolist()

    def trimmed_mean(self,
                    updates: List[np.ndarray],
                    trim_ratio: float = 0.2) -> np.ndarray:
        """
        Trimmed mean aggregation: Exclude extreme updates.

        Args:
            updates: List of client updates
            trim_ratio: Fraction to trim from each extreme

        Returns:
            Aggregated update
        """
        # Stack updates
        stacked = np.stack(updates, axis=0)

        # Compute norms for each update
        norms = np.linalg.norm(stacked.reshape(len(updates), -1), axis=1)

        # Sort by norm
        sorted_indices = np.argsort(norms)

        # Trim from both ends
        n_trim = int(len(updates) * trim_ratio)
        start_idx = n_trim
        end_idx = len(updates) - n_trim

        selected_indices = sorted_indices[start_idx:end_idx]

        # Average selected updates
        selected = stacked[selected_indices]
        aggregated = np.mean(selected, axis=0)

        return aggregated

    def coordinate_wise_trimmed_mean(self,
                                    updates: List[np.ndarray],
                                    trim_ratio: float = 0.2) -> List[np.ndarray]:
        """
        Coordinate-wise trimmed mean (stronger defense).

        Args:
            updates: List of client updates
            trim_ratio: Fraction to trim

        Returns:
            Aggregated update (list of arrays)
        """
        # Stack updates
        stacked = np.stack(updates, axis=0)  # (n_clients, n_params)

        # For each parameter, trim extreme values
        aggregated = []
        n_trim = int(len(updates) * trim_ratio)

        for param_idx in range(stacked.shape[1]):
            param_values = stacked[:, param_idx]

            # Sort and trim
            sorted_indices = np.argsort(param_values)
            selected_indices = sorted_indices[n_trim:-n_trim]

            aggregated.append(np.mean(param_values[selected_indices]))

        return aggregated
