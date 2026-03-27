"""Data module for loading and preprocessing."""

from .phishing_dataset import (
    PhishingEmailDataset,
    load_bank_data,
    create_dataloader,
    get_tokenizer
)

__all__ = [
    'PhishingEmailDataset',
    'load_bank_data',
    'create_dataloader',
    'get_tokenizer'
]
