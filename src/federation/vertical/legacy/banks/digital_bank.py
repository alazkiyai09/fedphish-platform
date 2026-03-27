"""
Digital Bank implementation.

App-focused bank with high smishing attacks.
"""

import yaml
from .base_bank import BaseBank, BankProfile
from ..data import load_bank_data, PhishingEmailDataset, get_tokenizer
from torch.utils.data import DataLoader


class DigitalBank(BaseBank):
    """
    Digital Bank: App-first, smishing-heavy.

    Characteristics:
    - 50K samples
    - High smishing (35%)
    - App spoofing attacks
    - Good data quality (0.88)
    - Digital-only channels
    - Newer attack vectors (+0.10 temporal shift)
    """

    def __init__(self, data_path: str, config: Dict = None):
        """Initialize Digital Bank."""
        if config is None:
            with open('config/bank_profiles.yaml', 'r') as f:
                config = yaml.safe_load(f)

        profile = BankProfile(
            name='Digital Bank',
            bank_type='app_first',
            n_samples=config['digital_bank']['n_samples'],
            phishing_distribution=config['digital_bank']['phishing_distribution'],
            data_quality=config['digital_bank']['data_quality'],
            languages=config['digital_bank']['languages'],
            temporal_shift=config['digital_bank']['temporal_shift'],
            geographic_scope=config['digital_bank']['geographic_scope'],
            customer_segments=config['digital_bank']['customer_segments']
        )

        super().__init__(profile, data_path)

    def load_data(self, split: str = 'train') -> PhishingEmailDataset:
        """Load Digital Bank's phishing dataset."""
        emails, labels, metadata = load_bank_data(
            bank_name='digital_bank',
            data_path=self.data_path,
            split=split
        )

        # Add digital characteristics
        for meta in metadata:
            meta['channel'] = 'mobile_app'
            meta['customer_type'] = 'digital_native'

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
        """Create DataLoader for Digital Bank."""
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
