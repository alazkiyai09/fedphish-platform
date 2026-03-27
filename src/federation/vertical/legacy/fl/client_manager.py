"""
Client manager for coordinating multiple bank clients.

Manages 5 bank clients for federated training.
"""

from typing import Dict, List, Callable
from dataclasses import dataclass

from ..banks import GlobalBank, RegionalBank, DigitalBank, CreditUnion, InvestmentBank
from .flower_client import PhishingDetectionClient, create_client


@dataclass
class ClientConfig:
    """Configuration for all bank clients."""
    n_clients: int = 5
    data_path: str = 'data/bank_datasets'
    privacy_mechanism: str = 'none'
    epsilon: float = 1.0


class ClientManager:
    """
    Manages multiple bank clients for federated learning.

    Creates and coordinates 5 different bank clients with
    non-IID data distributions.
    """

    def __init__(self, n_clients: int = 5):
        """
        Initialize client manager.

        Args:
            n_clients: Number of bank clients (default 5)
        """
        self.n_clients = n_clients
        self.clients: Dict[str, PhishingDetectionClient] = {}
        self.banks: Dict[str, BaseBank] = {}

    def create_all_clients(self,
                          data_path: str,
                          privacy_mechanism: str = 'none',
                          epsilon: float = 1.0) -> None:
        """
        Create all 5 bank clients.

        Args:
            data_path: Path to data directory
            privacy_mechanism: Type of privacy mechanism
            epsilon: Privacy budget for DP
        """
        # Create all banks
        self.banks['global_bank'] = GlobalBank(data_path=data_path)
        self.banks['regional_bank'] = RegionalBank(data_path=data_path)
        self.banks['digital_bank'] = DigitalBank(data_path=data_path)
        self.banks['credit_union'] = CreditUnion(data_path=data_path)
        self.banks['investment_bank'] = InvestmentBank(data_path=data_path)

        # Create clients
        for bank_name, bank in self.banks.items():
            self.clients[bank_name] = create_client(
                bank=bank,
                privacy_mechanism=privacy_mechanism,
                epsilon=epsilon
            )

    def get_client(self, bank_name: str) -> PhishingDetectionClient:
        """
        Get a specific client.

        Args:
            bank_name: Name of the bank

        Returns:
            PhishingDetectionClient
        """
        return self.clients.get(bank_name)

    def get_all_clients(self) -> Dict[str, PhishingDetectionClient]:
        """Get all clients."""
        return self.clients

    def get_client_fn(self,
                      privacy_mechanism: str = 'none',
                      epsilon: float = 1.0) -> Callable:
        """
        Get client function for Flower simulation.

        Args:
            privacy_mechanism: Type of privacy mechanism
            epsilon: Privacy budget

        Returns:
            Client function compatible with Flower
        """
        # Create clients if not exists
        if not self.clients:
            self.create_all_clients(
                data_path='data/bank_datasets',
                privacy_mechanism=privacy_mechanism,
                epsilon=epsilon
            )

        def client_fn(cid: str) -> PhishingDetectionClient:
            """
            Client function for Flower.

            Args:
                cid: Client ID (bank name)

            Returns:
                PhishingDetectionClient
            """
            # Map cid to bank name
            bank_names = list(self.banks.keys())
            bank_name = bank_names[int(cid) % len(bank_names)]

            return self.clients[bank_name]

        return client_fn

    def get_n_samples_per_bank(self) -> Dict[str, int]:
        """Get number of training samples per bank."""
        n_samples = {}

        for bank_name, client in self.clients.items():
            n_samples[bank_name] = len(client.train_loader.dataset)

        return n_samples

    def get_attack_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get phishing attack distribution for each bank."""
        distributions = {}

        for bank_name, bank in self.banks.items():
            distributions[bank_name] = bank.get_attack_distribution()

        return distributions

    def get_data_qualities(self) -> Dict[str, float]:
        """Get data quality scores for each bank."""
        qualities = {}

        for bank_name, bank in self.banks.items():
            qualities[bank_name] = bank.get_data_quality()

        return qualities
