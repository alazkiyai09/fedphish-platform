"""
Unit tests for FL protocol and defenses.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from fl import ClientManager
from attacks import MaliciousClient, DefenseMechanism
from evaluation import compute_metrics


class TestFLProtocol:
    """Test Flower federated learning protocol."""

    def test_client_manager_initialization(self):
        """Test client manager initialization."""
        manager = ClientManager(n_clients=5)

        assert manager.n_clients == 5
        assert isinstance(manager.clients, dict)
        assert len(manager.clients) == 0  # Not created yet

    def test_create_all_clients(self):
        """Test creating all bank clients."""
        manager = ClientManager(n_clients=5)
        manager.create_all_clients(
            data_path='data/bank_datasets',
            privacy_mechanism='none'
        )

        assert len(manager.clients) == 5
        assert 'global_bank' in manager.clients
        assert 'regional_bank' in manager.clients

    def test_get_n_samples_per_bank(self):
        """Test getting sample counts."""
        manager = ClientManager(n_clients=5)
        manager.create_all_clients(
            data_path='data/bank_datasets',
            privacy_mechanism='none'
        )

        n_samples = manager.get_n_samples_per_bank()

        assert len(n_samples) == 5
        assert n_samples['global_bank'] == 100000 * 0.8  # 80% for training
        assert n_samples['credit_union'] == 15000 * 0.8


class TestMaliciousClient:
    """Test malicious client implementation."""

    def test_label_flip_attack(self):
        """Test label flipping attack."""
        malicious = MaliciousClient(attack_type='label_flip', poisoning_rate=0.1)

        gradients = [
            torch.randn(10, 100) * 0.1,
            torch.randn(10, 100) * 0.1,
            torch.randn(10, 100) * 0.1
        ]

        poisoned = malicious.poison_gradients(gradients)

        assert len(poisoned) == 3
        # Gradients should be inverted
        for orig, poison in zip(gradients, poisoned):
            assert torch.allclose(poison, -orig * 0.5, atol=1e-6)

    def test_byzantine_attack(self):
        """Test Byzantine attack."""
        malicious = MaliciousClient(attack_type='byzantine', poisoning_rate=0.1)

        updates = [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]

        poisoned = malicious.poison_updates(updates)

        assert len(poisoned) == 3
        # Updates should be modified
        for orig, poison in zip(updates, poisoned):
            assert not np.array_equal(orig, poison)

    def test_backdoor_attack(self):
        """Test backdoor attack."""
        malicious = MaliciousClient(attack_type='backdoor', poisoning_rate=0.1)

        updates = [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]

        poisoned = malicious.poison_updates(updates)

        assert len(poisoned) == 3


class TestDefenses:
    """Test defense mechanisms."""

    def test_krum_defense(self):
        """Test Krum defense mechanism."""
        defense = DefenseMechanism(n_clients=5, n_malicious=1)

        # Create fake updates
        updates = [
            np.random.randn(100),
            np.random.randn(100) * 1.5,  # Malicious: larger magnitude
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]

        # Krum should exclude the malicious client
        best_idx = defense.krum(updates, n_malicious=1)

        # Should not select the malicious one (index 1)
        assert best_idx != 1

    def test_multi_krum(self):
        """Test Multi-Krum defense."""
        defense = DefenseMechanism(n_clients=5, n_malicious=1)

        updates = [
            np.random.randn(100),
            np.random.randn(100) * 2.0,  # Malicious
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]

        selected = defense.multi_krum(updates, n_malicious=1)

        # Should exclude malicious client
        assert 1 not in selected
        assert len(selected) == 4

    def test_trimmed_mean(self):
        """Test trimmed mean aggregation."""
        defense = DefenseMechanism(n_clients=5, n_malicious=1)

        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),  # Malicious: much larger
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0])
        ]

        aggregated = defense.trimmed_mean(updates, trim_ratio=0.2)

        # Should exclude extreme values (the malicious one)
        # Trim 20%: 5 clients * 0.2 = 1 client trimmed from each side
        # The malicious update with values [10, 20, 30] should be excluded
        expected = np.mean([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])

        assert np.allclose(aggregated, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
