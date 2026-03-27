"""
Investment Bank implementation.

High-value bank with sophisticated whaling attacks.
"""

import yaml
from .base_bank import BaseBank, BankProfile
from ..data import load_bank_data, PhishingEmailDataset, get_tokenizer
from torch.utils.data import DataLoader


class InvestmentBank(BaseBank):
    """
    Investment Bank: High-value, sophisticated attacks.

    Characteristics:
    - 10K samples
    - Very high whaling (35%) - C-level targets
    - High data quality (0.92) - sophisticated security
    - Business email compromise
    - Cutting edge attacks (+0.15 temporal shift)
    """

    def __init__(self, data_path: str, config: Dict = None):
        """Initialize Investment Bank."""
        if config is None:
            with open('config/bank_profiles.yaml', 'r') as f:
                config = yaml.safe_load(f)

        profile = BankProfile(
            name='Investment Bank',
            bank_type='high_value',
            n_samples=config['investment_bank']['n_samples'],
            phishing_distribution=config['investment_bank']['phishing_distribution'],
            data_quality=config['investment_bank']['data_quality'],
            languages=config['investment_bank']['languages'],
            temporal_shift=config['investment_bank']['temporal_shift'],
            geographic_scope=config['investment_bank']['geographic_scope'],
            customer_segments=config['investment_bank']['customer_segments']
        )

        super().__init__(profile, data_path)

    def load_data(self, split: str = 'train') -> PhishingEmailDataset:
        """Load Investment Bank's phishing dataset."""
        emails, labels, metadata = load_bank_data(
            bank_name='investment_bank',
            data_path=self.data_path,
            split=split
        )

        # Add investment bank characteristics
        for meta in metadata:
            meta['target_value'] = 'high'
            meta['attack_sophistication'] = 'advanced'

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
        """Create DataLoader for Investment Bank."""
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
