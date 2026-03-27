"""Integration tests for federated training."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedphish.client.model import create_model
from fedphish.client.trainer import FedPhishClient


class TestFederatedTraining:
    """Integration tests for federated training loop."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Simple synthetic data
        texts = [
            "Urgent verify account",
            "Meeting tomorrow",
            "Free gift click here",
            "Your statement ready",
        ] * 50  # 200 samples

        labels = [1, 0, 1, 0] * 50  # Alternating labels

        return texts, labels

    @pytest.fixture
    def data_loaders(self, sample_data):
        """Create data loaders."""
        texts, labels = sample_data

        # Create simple dataset (just using indices as placeholder)
        # In real test, would tokenize texts
        dataset = TensorDataset(
            torch.randint(0, 1000, (len(texts), 128)),  # Fake input_ids
            torch.ones(len(texts), 128).long(),  # Fake attention_mask
            torch.tensor(labels),
        )

        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader

    @pytest.fixture
    def model(self):
        """Create model."""
        device = "cpu"  # Use CPU for tests
        model = create_model(
            model_name="distilbert-base-uncased",
            num_labels=2,
            lora_rank=4,  # Small rank for testing
            device=device,
        )
        return model

    def test_client_initialization(self, model, data_loaders):
        """Test client initialization."""
        train_loader, val_loader = data_loaders

        client = FedPhishClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            privacy_level=1,  # Simplest for testing
            epsilon=1.0,
            delta=1e-5,
            max_gradient_norm=1.0,
            enable_zk_proofs=False,  # Disable for testing
            device="cpu",
        )

        assert client is not None
        assert client.model is not None

    def test_get_parameters(self, model, data_loaders):
        """Test getting model parameters."""
        train_loader, val_loader = data_loaders

        client = FedPhishClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            privacy_level=1,
            device="cpu",
        )

        params = client.get_parameters({})

        assert params is not None
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters(self, model, data_loaders):
        """Test setting model parameters."""
        train_loader, val_loader = data_loaders

        client = FedPhishClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            privacy_level=1,
            device="cpu",
        )

        # Get parameters
        params = client.get_parameters({})

        # Set parameters back
        client.set_parameters(params, {})

        # Should not error
        assert True

    def test_client_fit(self, model, data_loaders):
        """Test client training (fit)."""
        train_loader, val_loader = data_loaders

        client = FedPhishClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            privacy_level=1,
            device="cpu",
        )

        # Get initial parameters
        initial_params = client.get_parameters({})

        # Train for 1 epoch
        updated_params, num_samples, metrics = client.fit(
            parameters=initial_params,
            config={"local_epochs": 1, "learning_rate": 2e-5, "batch_size": 16},
        )

        # Check results
        assert updated_params is not None
        assert num_samples > 0
        assert "accuracy" in metrics
        assert "loss" in metrics

    def test_client_evaluate(self, model, data_loaders):
        """Test client evaluation."""
        train_loader, val_loader = data_loaders

        client = FedPhishClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            privacy_level=1,
            device="cpu",
        )

        params = client.get_parameters({})

        # Evaluate
        loss, num_samples, metrics = client.evaluate(
            parameters=params,
            config={},
        )

        # Check results
        assert loss >= 0
        assert num_samples > 0
        assert "accuracy" in metrics


class TestFederatedRound:
    """Test a simulated federated round."""

    @pytest.fixture
    def client_models(self):
        """Create multiple client models."""
        models = []
        for i in range(3):
            model = create_model(
                model_name="distilbert-base-uncased",
                num_labels=2,
                lora_rank=4,
                device="cpu",
            )
            models.append(model)
        return models

    @pytest.fixture
    def sample_data(self):
        """Create sample data for each client."""
        all_loaders = []
        for i in range(3):
            # Different data for each client
            texts = [f"Client {i} text {j}" for j in range(100)]
            labels = [j % 2 for j in range(100)]

            dataset = TensorDataset(
                torch.randint(0, 1000, (100, 128)),
                torch.ones(100, 128).long(),
                torch.tensor(labels),
            )
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            all_loaders.append(loader)
        return all_loaders

    def test_federated_round(self, client_models, sample_data):
        """Test one round of federated learning."""
        # Create clients
        clients = []
        for model, loader in zip(client_models, sample_data):
            client = FedPhishClient(
                model=model,
                train_loader=loader,
                val_loader=loader,
                privacy_level=1,
                device="cpu",
            )
            clients.append(client)

        # Get global parameters
        global_params = clients[0].get_parameters({})

        # Client training
        client_updates = []
        for client in clients:
            params, _, _ = client.fit(
                parameters=global_params,
                config={"local_epochs": 1, "learning_rate": 2e-5, "batch_size": 16},
            )
            client_updates.append(params)

        # Aggregate (simple average)
        aggregated = []
        for i in range(len(global_params)):
            param_list = [update[i] for update in client_updates]
            aggregated.append(np.mean(param_list, axis=0))

        # Check aggregation
        assert len(aggregated) == len(global_params)
        assert aggregated[0].shape == global_params[0].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
