"""
Phishing email dataset loader.

Loads and preprocesses phishing email datasets for all banks.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


class PhishingEmailDataset(Dataset):
    """
    Dataset for phishing email classification.

    Each sample consists of:
    - Email text (subject + body)
    - Binary label (0: safe, 1: phishing)
    - Metadata (attack type, bank source, etc.)
    """

    def __init__(self,
                 emails: List[str],
                 labels: List[int],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 metadata: List[Dict] = None):
        """
        Initialize dataset.

        Args:
            emails: List of email texts
            labels: Binary labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            metadata: Optional metadata for each email
        """
        self.emails = emails
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = metadata or [{}] * len(emails)

    def __len__(self) -> int:
        return len(self.emails)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, label, metadata
        """
        email_text = self.emails[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            email_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'metadata': self.metadata[idx]
        }


def load_bank_data(bank_name: str,
                   data_path: str,
                   split: str = 'train') -> Tuple[List[str], List[int], List[Dict]]:
    """
    Load data for a specific bank.

    Args:
        bank_name: Name of the bank
        data_path: Path to data directory
        split: 'train' or 'test'

    Returns:
        Tuple of (emails, labels, metadata)
    """
    # For now, create synthetic data
    # In production, this would load real phishing datasets

    import random
    random.seed(hash(bank_name + split) % 2**32)

    n_samples = 1000 if split == 'train' else 200

    emails = []
    labels = []
    metadata = []

    # Simulated email templates
    phishing_templates = [
        "URGENT: Your account will be suspended. Click here to verify: {url}",
        "Dear Customer, we detected suspicious activity. Confirm: {url}",
        "Invoice attached. Please review immediately: {url}",
        "You have won! Claim your prize now: {url}",
        "Security alert: Update your password: {url}",
    ]

    safe_templates = [
        "Your monthly statement is ready.",
        "Thank you for your recent transaction.",
        "Your appointment is confirmed.",
        "Your balance is ${amount}.",
        "Welcome to our service!",
    ]

    for i in range(n_samples):
        # Simulate phishing distribution
        is_phishing = random.random() < 0.4  # 40% phishing

        if is_phishing:
            template = random.choice(phishing_templates)
            url = f"http://suspicious-site-{random.randint(1000,9999)}.com"
            email = template.format(url=url)
            label = 1
            attack_type = random.choice(['spear_phishing', 'generic_phishing', 'smishing'])
        else:
            template = random.choice(safe_templates)
            email = template.format(amount=f"{random.randint(100,10000):,.2f}")
            label = 0
            attack_type = 'none'

        emails.append(email)
        labels.append(label)
        metadata.append({
            'bank': bank_name,
            'attack_type': attack_type,
            'sample_id': f"{bank_name}_{split}_{i}"
        })

    return emails, labels, metadata


def create_dataloader(bank_name: str,
                      data_path: str,
                      tokenizer: AutoTokenizer,
                      batch_size: int = 32,
                      split: str = 'train') -> torch.utils.data.DataLoader:
    """
    Create DataLoader for a bank.

    Args:
        bank_name: Name of the bank
        data_path: Path to data directory
        tokenizer: Tokenizer
        batch_size: Batch size
        split: 'train' or 'test'

    Returns:
        DataLoader
    """
    emails, labels, metadata = load_bank_data(bank_name, data_path, split)

    dataset = PhishingEmailDataset(
        emails=emails,
        labels=labels,
        tokenizer=tokenizer,
        max_length=512,
        metadata=metadata
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train')
    )


def get_tokenizer(model_name: str = 'distilbert-base-uncased') -> AutoTokenizer:
    """Get tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)
