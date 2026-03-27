"""
Regional Bank implementation.

Local bank with high spear phishing and localized attacks.
"""

import yaml
from .base_bank import BaseBank, BankProfile
from ..data import load_bank_data, PhishingEmailDataset, get_tokenizer
from torch.utils.data import DataLoader


class RegionalBank(BaseBank):
    """
    Regional Bank: Local, targeted phishing.

    Characteristics:
    - 30K samples
    - High spear phishing (40%)
    - Local brand impersonation
    - Moderate data quality (0.82)
    - Primarily English
    - Slight temporal shift (+0.05)
    """

    def __init__(self, data_path: str, config: Dict = None):
        """Initialize Regional Bank."""
        if config is None:
            with open('config/bank_profiles.yaml', 'r') as f:
                config = yaml.safe_load(f)

        profile = BankProfile(
            name='Regional Bank',
            bank_type='local',
            n_samples=config['regional_bank']['n_samples'],
            phishing_distribution=config['regional_bank']['phishing_distribution'],
            data_quality=config['regional_bank']['data_quality'],
            languages=config['regional_bank']['languages'],
            temporal_shift=config['regional_bank']['temporal_shift'],
            geographic_scope=config['regional_bank']['geographic_scope'],
            customer_segments=config['regional_bank']['customer_segments']
        )

        super().__init__(profile, data_path)

    def load_data(self, split: str = 'train') -> PhishingEmailDataset:
        """Load Regional Bank's phishing dataset."""
        emails, labels, metadata = load_bank_data(
            bank_name='regional_bank',
            data_path=self.data_path,
            split=split
        )

        # Add regional characteristics
        for meta in metadata:
            meta['geographic_scope'] = 'north_america'
            meta['market_focus'] = 'regional'

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
        """Create DataLoader for Regional Bank."""
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
