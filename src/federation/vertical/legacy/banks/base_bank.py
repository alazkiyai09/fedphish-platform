"""
Base bank class for cross-bank federated phishing detection.

All bank types inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class BankProfile:
    """Bank configuration profile."""
    name: str
    bank_type: str
    n_samples: int
    phishing_distribution: Dict[str, float]
    data_quality: float
    languages: List[str]
    temporal_shift: float
    geographic_scope: str
    customer_segments: List[str]


class BaseBank(ABC):
    """
    Abstract base class for all banks.

    Each bank has:
    - Unique data characteristics (non-IID)
    - Different phishing type distributions
    - Different data quality
    - Different sample volumes
    """

    def __init__(self, profile: BankProfile, data_path: str):
        """
        Initialize bank.

        Args:
            profile: Bank configuration profile
            data_path: Path to bank's data directory
        """
        self.profile = profile
        self.data_path = data_path
        self.train_dataset = None
        self.test_dataset = None

    @abstractmethod
    def load_data(self, split: str = 'train') -> Dataset:
        """
        Load bank's phishing dataset.

        Args:
            split: 'train' or 'test'

        Returns:
            PyTorch Dataset
        """
        pass

    @abstractmethod
    def create_dataloader(self, batch_size: int = 32,
                         split: str = 'train') -> DataLoader:
        """
        Create DataLoader for bank's data.

        Args:
            batch_size: Batch size
            split: 'train' or 'test'

        Returns:
            PyTorch DataLoader
        """
        pass

    def get_profile(self) -> BankProfile:
        """Get bank's profile configuration."""
        return self.profile

    def get_n_samples(self, split: str = 'train') -> int:
        """Get number of samples for this bank."""
        if split == 'train':
            return int(self.profile.n_samples * 0.8)
        else:
            return int(self.profile.n_samples * 0.2)

    def get_attack_distribution(self) -> Dict[str, float]:
        """Get distribution of phishing attack types."""
        return self.profile.phishing_distribution

    def get_data_quality(self) -> float:
        """
        Get data quality score (label accuracy).

        Returns:
            Float between 0 and 1
        """
        return self.profile.data_quality

    def get_phishing_types(self) -> List[str]:
        """Get list of phishing attack types seen by this bank."""
        return list(self.profile.phishing_distribution.keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.profile.name}', n_samples={self.profile.n_samples})"
