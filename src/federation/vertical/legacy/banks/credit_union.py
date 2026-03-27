"""
Credit Union implementation.

Member-focused bank with trust exploitation attacks.
"""

import yaml
from .base_bank import BaseBank, BankProfile
from ..data import load_bank_data, PhishingEmailDataset, get_tokenizer
from torch.utils.data import DataLoader


class CreditUnion(BaseBank):
    """
    Credit Union: Member-focused, trust-based attacks.

    Characteristics:
    - 15K samples (smallest)
    - High trust exploitation (15%)
    - Lower data quality (0.75) - smaller security team
    - Local community focus
    - Slightly older attacks (-0.05 temporal shift)
    """

    def __init__(self, data_path: str, config: Dict = None):
        """Initialize Credit Union."""
        if config is None:
            with open('config/bank_profiles.yaml', 'r') as f:
                config = yaml.safe_load(f)

        profile = BankProfile(
            name='Credit Union',
            bank_type='member_focused',
            n_samples=config['credit_union']['n_samples'],
            phishing_distribution=config['credit_union']['phishing_distribution'],
            data_quality=config['credit_union']['data_quality'],
            languages=config['credit_union']['languages'],
            temporal_shift=config['credit_union']['temporal_shift'],
            geographic_scope=config['credit_union']['geographic_scope'],
            customer_segments=config['credit_union']['customer_segments']
        )

        super().__init__(profile, data_path)

    def load_data(self, split: str = 'train') -> PhishingEmailDataset:
        """Load Credit Union's phishing dataset."""
        emails, labels, metadata = load_bank_data(
            bank_name='credit_union',
            data_path=self.data_path,
            split=split
        )

        # Add credit union characteristics
        for meta in metadata:
            meta['membership_based'] = True
            meta['community_focus'] = 'local'

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
        """Create DataLoader for Credit Union."""
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
