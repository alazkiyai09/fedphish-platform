"""Federated Learning infrastructure: strategies, client, server, privacy."""

from .client import FedPhishClient, create_client
from .server import FedPhishServer, start_server
from .strategies import get_strategy, FedAvgStrategy, FedProxStrategy
from .privacy import (
    DifferentialPrivacy,
    SecureAggregation,
    add_dp_noise,
    clip_gradients
)

__all__ = [
    "FedPhishClient",
    "create_client",
    "FedPhishServer",
    "start_server",
    "get_strategy",
    "FedAvgStrategy",
    "FedProxStrategy",
    "DifferentialPrivacy",
    "SecureAggregation",
    "add_dp_noise",
    "clip_gradients",
]
