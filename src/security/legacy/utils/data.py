"""Data loading and partitioning utilities."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def load_phishing_data(
    num_samples: int = 10000,
    num_features: int = 100,
    num_classes: int = 2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load or generate phishing dataset.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        num_classes: Number of classes (2 for binary classification)
        train_split: Training set ratio
        val_split: Validation set ratio
        test_split: Test set ratio
        seed: Random seed

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate synthetic phishing data
    # In real use case, load from CSV file
    X = torch.randn(num_samples, num_features)

    # Generate labels with some pattern
    # Class 0: legitimate, Class 1: phishing
    y = torch.zeros(num_samples, dtype=torch.long)

    # Create pattern: phishing samples have certain feature characteristics
    phishing_indices = np.random.choice(num_samples, size=num_samples // 2, replace=False)
    y[phishing_indices] = 1

    # Add some signal to features for phishing samples
    X[phishing_indices, :10] += 1.0  # First 10 features indicate phishing

    # Shuffle data
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    # Split data
    num_train = int(num_samples * train_split)
    num_val = int(num_samples * val_split)

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_val = X[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]

    X_test = X[num_train + num_val:]
    y_test = y[num_train + num_val:]

    logger.info(
        f"Loaded phishing dataset: "
        f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def partition_data(
    X: torch.Tensor,
    y: torch.Tensor,
    num_clients: int,
    distribution: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Partition data among clients.

    Args:
        X: Data features
        y: Data labels
        num_clients: Number of clients
        distribution: Distribution type ("iid", "non_iid", "label_skew")
        alpha: Dirichlet concentration parameter (for non-iid)
        seed: Random seed

    Returns:
        Dictionary mapping client_id to (X, y) tuple
    """
    np.random.seed(seed)
    num_samples = len(X)

    partitions = {i: ([], []) for i in range(num_clients)}

    if distribution == "iid":
        # Random partition
        indices = np.random.permutation(num_samples)
        chunk_size = num_samples // num_clients

        for client_id in range(num_clients):
            start = client_id * chunk_size
            end = start + chunk_size if client_id < num_clients - 1 else num_samples

            client_indices = indices[start:end]
            partitions[client_id] = (X[client_indices], y[client_indices])

    elif distribution == "non_iid":
        # Dirichlet-based non-IID partition
        num_classes = len(y.unique())
        label_distribution = np.random.dirichlet([alpha] * num_clients, size=num_classes)

        for label in range(num_classes):
            label_indices = np.where(y.numpy() == label)[0]
            np.random.shuffle(label_indices)

            start = 0
            for client_id in range(num_clients):
                end = start + int(label_distribution[label, client_id] * len(label_indices))

                if client_id == num_clients - 1:
                    end = len(label_indices)

                client_label_indices = label_indices[start:end]
                partitions[client_id][0].extend(client_label_indices)
                partitions[client_id][1].extend([label] * len(client_label_indices))

                start = end

        # Convert to tensors
        for client_id in range(num_clients):
            indices = partitions[client_id][0]
            partitions[client_id] = (X[indices], y[indices])

    elif distribution == "label_skew":
        # Each client gets primarily one label
        num_classes = len(y.unique())

        for client_id in range(num_clients):
            # Assign primary label to client
            primary_label = client_id % num_classes

            # Get all samples with this label
            label_indices = np.where(y.numpy() == primary_label)[0]

            # Add some samples from other labels (20%)
            other_label_indices = np.where(y.numpy() != primary_label)[0]
            num_other = min(len(other_label_indices) // num_clients, len(label_indices) // 4)
            selected_other = np.random.choice(other_label_indices, size=num_other, replace=False)

            client_indices = np.concatenate([label_indices, selected_other])
            partitions[client_id] = (X[client_indices], y[client_indices])

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    # Log partition info
    for client_id in range(num_clients):
        X_client, y_client = partitions[client_id]
        unique_labels = y_client.unique()
        logger.debug(
            f"Client {client_id}: {len(X_client)} samples, "
            f"Labels: {unique_labels.numpy()}"
        )

    return partitions
