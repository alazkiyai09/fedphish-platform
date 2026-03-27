"""Public simulator surface for the FedPhish dashboard backend."""

from __future__ import annotations

from src.dashboard.backend.core.simulator import BankState, FederatedSimulator

__all__ = ["BankState", "FederatedSimulator", "create_simulator"]


def create_simulator(scenario):
    """Factory used by scripts and tests to instantiate the real simulator."""
    return FederatedSimulator(scenario)
