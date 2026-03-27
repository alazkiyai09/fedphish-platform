"""Data partitioning for federated learning (IID, non-IID, label skew)."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import dirichlet

logger = logging.getLogger(__name__)


class DataPartitioner:
    """Partition data for federated learning scenarios."""

    def __init__(self, num_clients: int, seed: int = 42):
        """
        Initialize partitioner.

        Args:
            num_clients: Number of clients (banks)
            seed: Random seed
        """
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        partition_type: str = "iid",
        **kwargs
    ) -> Dict[int, np.ndarray]:
        """
        Partition data according to specified strategy.

        Args:
            X: Features
            y: Labels
            partition_type: Type of partition (iid, non_iid, label_skew)
            **kwargs: Additional arguments for specific partition types

        Returns:
            Dictionary mapping client_id to indices
        """
        num_samples = len(X)

        if partition_type == "iid":
            return self._partition_iid(num_samples, **kwargs)
        elif partition_type == "non_iid":
            return self._partition_non_iid(y, **kwargs)
        elif partition_type == "label_skew":
            return self._partition_label_skew(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

    def _partition_iid(
        self,
        num_samples: int,
        shuffle: bool = True,
        stratify: bool = False,
        y: np.ndarray = None
    ) -> Dict[int, np.ndarray]:
        """
        Partition data IID across clients.

        Args:
            num_samples: Total number of samples
            shuffle: Whether to shuffle data
            stratify: Whether to stratify by label
            y: Labels (required if stratify=True)

        Returns:
            Dictionary mapping client_id to indices
        """
        indices = np.arange(num_samples)

        if shuffle:
            if stratify and y is not None:
                # Stratified shuffle
                indices = self._stratified_shuffle(indices, y)
            else:
                np.random.shuffle(indices)

        # Split indices among clients
        samples_per_client = num_samples // self.num_clients
        partitions = {}

        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:
                # Last client gets remaining samples
                end_idx = num_samples
            else:
                end_idx = (i + 1) * samples_per_client

            partitions[i] = indices[start_idx:end_idx]

        logger.info(f"IID partition: {[(k, len(v)) for k, v in partitions.items()]}")
        return partitions

    def _partition_non_iid(
        self,
        y: np.ndarray,
        alpha: float = 0.5,
        min_samples_per_client: int = 100
    ) -> Dict[int, np.ndarray]:
        """
        Partition data non-IID using Dirichlet distribution.

        Args:
            y: Labels
            alpha: Dirichlet concentration (lower = more skewed)
            min_samples_per_client: Minimum samples per client

        Returns:
            Dictionary mapping client_id to indices
        """
        num_classes = len(np.unique(y))
        num_samples = len(y)

        # Generate Dirichlet distribution for each label
        # shape: (num_classes, num_clients)
        proportions = dirichlet.rvs(alpha * np.ones(self.num_clients), size=num_classes, random_state=self.seed)

        partitions = {i: [] for i in range(self.num_clients)}

        for label in range(num_classes):
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)

            # Distribute this label across clients according to proportions
            start = 0
            for client_id in range(self.num_clients - 1):
                end = start + int(proportions[label, client_id] * len(label_indices))
                partitions[client_id].extend(label_indices[start:end])
                start = end
            # Last client gets remaining samples for this label
            partitions[self.num_clients - 1].extend(label_indices[start:])

        # Ensure minimum samples per client
        for client_id in range(self.num_clients):
            if len(partitions[client_id]) < min_samples_per_client:
                # Borrow from other clients (without duplication)
                needed = min_samples_per_client - len(partitions[client_id])
                for other_id in range(self.num_clients):
                    if other_id != client_id and len(partitions[other_id]) > min_samples_per_client:
                        # Transfer some samples from other client
                        available = min(needed, len(partitions[other_id]) - min_samples_per_client)
                        transfer = partitions[other_id][:available]
                        partitions[client_id].extend(transfer)
                        partitions[other_id] = partitions[other_id][available:]
                        needed -= available
                        if needed <= 0:
                            break

        # Convert to numpy arrays
        partitions = {k: np.array(v, dtype=np.int32) for k, v in partitions.items()}

        logger.info(f"Non-IID partition (alpha={alpha}): {[(k, len(v)) for k, v in partitions.items()]}")
        return partitions

    def _partition_label_skew(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labels_per_client: int = 2,
        min_samples_per_client: int = 100
    ) -> Dict[int, np.ndarray]:
        """
        Partition data with label skew (each client sees only few labels).

        Args:
            X: Features (not used, for consistency)
            y: Labels
            labels_per_client: Number of labels per client
            min_samples_per_client: Minimum samples per client

        Returns:
            Dictionary mapping client_id to indices
        """
        num_classes = len(np.unique(y))
        partitions = {i: [] for i in range(self.num_clients)}

        # Track assigned indices to avoid duplication
        assigned_indices = set()

        # Assign label combinations to clients
        for client_id in range(self.num_clients):
            # Determine which labels this client sees
            start_label = (client_id * labels_per_client) % num_classes
            client_labels = [(start_label + i) % num_classes for i in range(labels_per_client)]

            # Get samples with these labels that haven't been assigned yet
            for label in client_labels:
                label_indices = np.where(y == label)[0]
                # Filter out already assigned indices
                available_indices = [idx for idx in label_indices if idx not in assigned_indices]
                partitions[client_id].extend(available_indices)
                assigned_indices.update(available_indices)

        # Convert to numpy arrays
        partitions = {k: np.array(v, dtype=np.int32) for k, v in partitions.items()}

        logger.info(f"Label skew partition: {[(k, len(v)) for k, v in partitions.items()]}")
        return partitions

    def _stratified_shuffle(self, indices: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Shuffle indices while maintaining label distribution.

        Args:
            indices: Indices to shuffle
            y: Labels

        Returns:
            Shuffled indices
        """
        unique_labels = np.unique(y)
        shuffled_indices = []

        for label in unique_labels:
            label_indices = indices[y[indices] == label]
            np.random.shuffle(label_indices)
            shuffled_indices.extend(label_indices)

        return np.array(shuffled_indices, dtype=np.int32)


def partition_iid(
    data: np.ndarray,
    num_clients: int,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Convenience function to partition data IID.

    Args:
        data: Data array (just need length)
        num_clients: Number of clients
        seed: Random seed

    Returns:
        Dictionary mapping client_id to indices
    """
    partitioner = DataPartitioner(num_clients, seed)
    return partitioner._partition_iid(len(data), shuffle=True)


def partition_non_iid(
    data: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Convenience function to partition data non-IID.

    Args:
        data: Labels array
        num_clients: Number of clients
        alpha: Dirichlet concentration
        seed: Random seed

    Returns:
        Dictionary mapping client_id to indices
    """
    partitioner = DataPartitioner(num_clients, seed)
    return partitioner._partition_non_iid(data, alpha=alpha)


def partition_label_skew(
    data: np.ndarray,
    num_clients: int,
    labels_per_client: int = 2,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Convenience function to partition data with label skew.

    Args:
        data: Labels array
        num_clients: Number of clients
        labels_per_client: Labels per client
        seed: Random seed

    Returns:
        Dictionary mapping client_id to indices
    """
    partitioner = DataPartitioner(num_clients, seed)
    return partitioner._partition_label_skew(None, data, labels_per_client=labels_per_client)
