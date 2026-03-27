"""Tests for data partitioning."""

import numpy as np
import pytest
from src.data.partitioner import DataPartitioner


class TestDataPartitioner:
    """Test data partitioner."""

    def test_partition_iid(self):
        """Test IID partitioning."""
        partitioner = DataPartitioner(num_clients=10, seed=42)
        num_samples = 1000

        X = np.random.randn(num_samples, 10)
        y = np.random.randint(0, 2, num_samples)

        partitions = partitioner.partition(X, y, partition_type="iid")

        assert len(partitions) == 10
        total_samples = sum(len(v) for v in partitions.values())
        assert total_samples == num_samples

    def test_partition_non_iid(self):
        """Test non-IID partitioning."""
        partitioner = DataPartitioner(num_clients=10, seed=42)
        num_samples = 1000

        X = np.random.randn(num_samples, 10)
        y = np.random.randint(0, 2, num_samples)

        partitions = partitioner.partition(X, y, partition_type="non_iid", alpha=0.5)

        assert len(partitions) == 10
        total_samples = sum(len(v) for v in partitions.values())
        assert total_samples == num_samples

    def test_partition_label_skew(self):
        """Test label skew partitioning."""
        partitioner = DataPartitioner(num_clients=10, seed=42)
        num_samples = 1000

        X = np.random.randn(num_samples, 10)
        y = np.random.randint(0, 5, num_samples)

        partitions = partitioner.partition(
            X, y, partition_type="label_skew", labels_per_client=2
        )

        assert len(partitions) == 10
        total_samples = sum(len(v) for v in partitions.values())
        assert total_samples == num_samples
