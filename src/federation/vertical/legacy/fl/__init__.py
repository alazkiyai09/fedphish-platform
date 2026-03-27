"""
Update src/fl/__init__.py to fix imports
"""

from .flower_client import PhishingDetectionClient, create_client
from .flower_server import create_server, run_federated_simulation, weighted_average
from .strategy import FedProxStrategy, AdaptiveStrategy, SecureAggregationStrategy
from .client_manager import ClientManager, ClientConfig

__all__ = [
    'PhishingDetectionClient',
    'create_client',
    'create_server',
    'run_federated_simulation',
    'weighted_average',
    'FedProxStrategy',
    'AdaptiveStrategy',
    'SecureAggregationStrategy',
    'ClientManager',
    'ClientConfig'
]
