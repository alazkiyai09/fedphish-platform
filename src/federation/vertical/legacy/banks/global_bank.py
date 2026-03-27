"""
Global Bank implementation.

International bank with diverse phishing types and high volume.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
import yaml

from .base_bank import BaseBank, BankProfile
from ..data import load_bank_data, PhishingEmailDataset, get_tokenizer


class GlobalBank(BaseBank):
    """
    Global Bank: International, diverse phishing.

    Characteristics:
    - 100K samples (largest)
    - 7 different phishing attack types
    - High data quality (0.95)
    - Multiple languages
    - Baseline reference for temporal shift
    """

    def __init__(self, data_path: str, config: Dict = None):
        """Initialize Global Bank."""
        # Load profile from config
        if config is None:
            with open('config/bank_profiles.yaml', 'r') as f:
                config = yaml.safe_load(f)

        profile = BankProfile(
            name='Global Bank',
            bank_type='international',
            n_samples=config['global_bank']['n_samples'],
            phishing_distribution=config['global_bank']['phishing_distribution'],
            data_quality=config['global_bank']['data_quality'],
            languages=config['global_bank']['languages'],
            temporal_shift=config['global_bank']['temporal_shift'],
            geographic_scope=config['global_bank']['geographic_scope'],
            customer_segments=config['global_bank']['customer_segments']
        )

        super().__init__(profile, data_path)

    def load_data(self, split: str = 'train') -> PhishingEmailDataset:
        """Load Global Bank's phishing dataset."""
        emails, labels, metadata = load_bank_data(
            bank_name='global_bank',
            data_path=self.data_path,
            split=split
        )

        # Add language diversity
        for i, meta in enumerate(metadata):
            meta['language'] = self.profile.languages[i % len(self.profile.languages)]
            meta['geographic_scope'] = 'global'

        tokenizer = get_tokenizer()
        dataset = PhishingEmailDataset(
            emails=emails,
            labels=labels,
            tokenizer=tokenizer,
            max_length=512,
            metadata=metadata
        )

        if split == 'train':
            self.train_dataset = dataset
        else:
            self.test_dataset = dataset

        return dataset

    def create_dataloader(self, batch_size: int = 32,
                         split: str = 'train') -> DataLoader:
        """Create DataLoader for Global Bank."""
        if split == 'train' and self.train_dataset is None:
            self.load_data('train')
        if split == 'test' and self.test_dataset is None:
            self.load_data('test')

        dataset = self.train_dataset if split == 'train' else self.test_dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2
        )
