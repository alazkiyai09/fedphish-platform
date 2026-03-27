"""Tests for FL strategies."""

import pytest
from src.fl.strategies import get_strategy, FedAvgStrategy, FedProxStrategy


class TestFLStrategies:
    """Test federated learning strategies."""

    def test_get_fedavg_strategy(self):
        """Test getting FedAvg strategy."""
        config = {
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.8,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
        }

        strategy = get_strategy("fedavg", config)

        assert strategy is not None
        assert isinstance(strategy, FedAvgStrategy)

    def test_get_fedprox_strategy(self):
        """Test getting FedProx strategy."""
        config = {
            "mu": 0.01,
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.8,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
        }

        strategy = get_strategy("fedprox", config)

        assert strategy is not None
        assert isinstance(strategy, FedProxStrategy)
        assert strategy.mu == 0.01

    def test_invalid_strategy(self):
        """Test invalid strategy name."""
        with pytest.raises(ValueError):
            get_strategy("invalid_strategy", {})
