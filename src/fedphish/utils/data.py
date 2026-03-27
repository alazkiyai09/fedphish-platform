"""
Data utilities for federated phishing detection.

Handles data loading, partitioning across banks, and preprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class PhishingDataset(Dataset):
    """PyTorch dataset for phishing detection."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize phishing dataset.

        Args:
            texts: List of text samples
            labels: List of labels (0=legitimate, 1=phishing)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BankDataPartitioner:
    """Partition data across multiple banks for federated learning."""

    def __init__(
        self,
        num_banks: int = 5,
        partition_strategy: str = "non-iid",  # 'iid', 'non-iid', 'imbalanced'
        alpha: float = 0.5,  # For Dirichlet distribution (non-IID)
        imbalance_factor: float = 0.3,  # For imbalanced partitions
        random_seed: int = 42,
    ):
        """
        Initialize bank data partitioner.

        Args:
            num_banks: Number of banks to partition data across
            partition_strategy: 'iid', 'non-iid', or 'imbalanced'
            alpha: Concentration parameter for Dirichlet (lower = more non-IID)
            imbalance_factor: Skew factor for imbalanced partitions
            random_seed: Random seed for reproducibility
        """
        self.num_banks = num_banks
        self.partition_strategy = partition_strategy
        self.alpha = alpha
        self.imbalance_factor = imbalance_factor
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.bank_assignments: Dict[int, List[int]] = {}

    def partition(
        self,
        texts: List[str],
        labels: List[int],
    ) -> Dict[int, Tuple[List[str], List[int]]]:
        """
        Partition data across banks.

        Args:
            texts: List of text samples
            labels: List of labels

        Returns:
            Dictionary mapping bank_id -> (bank_texts, bank_labels)
        """
        num_samples = len(texts)
        indices = np.arange(num_samples)

        if self.partition_strategy == "iid":
            return self._partition_iid(texts, labels, indices)
        elif self.partition_strategy == "non-iid":
            return self._partition_non_iid(texts, labels, indices)
        elif self.partition_strategy == "imbalanced":
            return self._partition_imbalanced(texts, labels, indices)
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")

    def _partition_iid(
        self,
        texts: List[str],
        labels: List[int],
        indices: np.ndarray,
    ) -> Dict[int, Tuple[List[str], List[int]]]:
        """Partition data IID across banks."""
        # Shuffle and split evenly
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.num_banks)

        bank_data = {}
        for bank_id, split in enumerate(splits):
            bank_texts = [texts[i] for i in split]
            bank_labels = [labels[i] for i in split]
            bank_data[bank_id] = (bank_texts, bank_labels)
            self.bank_assignments[bank_id] = split.tolist()

        logger.info(f"Partitioned data IID across {self.num_banks} banks")
        return bank_data

    def _partition_non_iid(
        self,
        texts: List[str],
        labels: List[int],
        indices: np.ndarray,
    ) -> Dict[int, Tuple[List[str], List[int]]]:
        """
        Partition data non-IID using Dirichlet distribution.

        Creates label skew where each bank has different class proportions.
        """
        bank_data = {}
        num_classes = len(set(labels))

        # Group samples by class
        class_indices = {}
        for label in set(labels):
            class_indices[label] = np.where(np.array(labels) == label)[0]

        # Assign samples to banks using Dirichlet
        for bank_id in range(self.num_banks):
            bank_indices = []
            proportions = np.random.dirichlet([self.alpha] * num_classes)

            for class_label, class_idx in class_indices.items():
                num_samples = max(1, int(len(class_idx) * proportions[class_label]))
                selected = np.random.choice(
                    class_idx, size=min(num_samples, len(class_idx)), replace=False
                )
                bank_indices.extend(selected)

            bank_texts = [texts[i] for i in bank_indices]
            bank_labels = [labels[i] for i in bank_indices]
            bank_data[bank_id] = (bank_texts, bank_labels)
            self.bank_assignments[bank_id] = bank_indices

        logger.info(
            f"Partitioned data non-IID (alpha={self.alpha}) across {self.num_banks} banks"
        )
        return bank_data

    def _partition_imbalanced(
        self,
        texts: List[str],
        labels: List[int],
        indices: np.ndarray,
    ) -> Dict[int, Tuple[List[str], List[int]]]:
        """
        Partition data with quantity imbalance.

        Some banks have much more data than others.
        """
        bank_data = {}
        num_samples = len(indices)

        # Create exponential distribution for bank sizes
        base_sizes = np.array(
            [self.imbalance_factor**i for i in range(self.num_banks)]
        )
        base_sizes = base_sizes / base_sizes.sum() * num_samples
        base_sizes = base_sizes.astype(int)

        # Ensure all samples are assigned
        base_sizes[-1] += num_samples - base_sizes.sum()

        # Split according to sizes
        np.random.shuffle(indices)
        start_idx = 0
        for bank_id in range(self.num_banks):
            end_idx = start_idx + base_sizes[bank_id]
            bank_indices = indices[start_idx:end_idx]
            start_idx = end_idx

            bank_texts = [texts[i] for i in bank_indices]
            bank_labels = [labels[i] for i in bank_indices]
            bank_data[bank_id] = (bank_texts, bank_labels)
            self.bank_assignments[bank_id] = bank_indices.tolist()

        logger.info(
            f"Partitioned data imbalanced (factor={self.imbalance_factor}) "
            f"across {self.num_banks} banks"
        )
        return bank_data


class PhishingDataLoader:
    """Load and preprocess phishing detection data."""

    def __init__(
        self,
        data_path: Union[str, Path],
        text_column: str = "text",
        label_column: str = "label",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Initialize data loader.

        Args:
            data_path: Path to data file (CSV, JSON, or JSONL)
            text_column: Name of text column
            label_column: Name of label column
            max_samples: Maximum number of samples to load (for testing)
            random_seed: Random seed
        """
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.label_column = label_column
        self.max_samples = max_samples
        self.random_seed = random_seed

    def load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load data from file.

        Returns:
            Tuple of (texts, labels)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == ".json":
            df = pd.read_json(self.data_path)
        elif self.data_path.suffix == ".jsonl":
            df = pd.read_json(self.data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        # Validate columns
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found")
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found")

        # Extract texts and labels
        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()

        # Limit samples if specified
        if self.max_samples and len(texts) > self.max_samples:
            indices = np.random.choice(
                len(texts), self.max_samples, replace=False
            ).tolist()
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]

        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels)}")

        return texts, labels

    def create_train_val_split(
        self,
        texts: List[str],
        labels: List[int],
        val_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Split data into train and validation sets.

        Args:
            texts: List of texts
            labels: List of labels
            val_size: Fraction of data for validation
            stratify: Whether to stratify the split

        Returns:
            Tuple of (train_texts, train_labels, val_texts, val_labels)
        """
        split_params = {"test_size": val_size, "random_state": self.random_seed}
        if stratify:
            split_params["stratify"] = labels

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, **split_params
        )

        logger.info(
            f"Train: {len(train_texts)} samples, Val: {len(val_texts)} samples"
        )

        return train_texts, train_labels, val_texts, val_labels

    def create_dataloaders(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create PyTorch DataLoader.

        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer
            batch_size: Batch size
            max_length: Maximum sequence length
            shuffle: Whether to shuffle data

        Returns:
            DataLoader
        """
        dataset = PhishingDataset(texts, labels, tokenizer, max_length)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to >0 for multi-worker loading
            pin_memory=True,
        )

        return dataloader


class BankSimulation:
    """Simulate multiple banks with different data distributions."""

    def __init__(
        self,
        data_path: Union[str, Path],
        num_banks: int = 5,
        partition_strategy: str = "non-iid",
        val_size: float = 0.2,
        tokenizer_name: str = "distilbert-base-uncased",
        random_seed: int = 42,
    ):
        """
        Initialize bank simulation.

        Args:
            data_path: Path to data file
            num_banks: Number of banks
            partition_strategy: Data partitioning strategy
            val_size: Validation split size
            tokenizer_name: HuggingFace tokenizer name
            random_seed: Random seed
        """
        self.data_path = Path(data_path)
        self.num_banks = num_banks
        self.partition_strategy = partition_strategy
        self.val_size = val_size
        self.tokenizer_name = tokenizer_name
        self.random_seed = random_seed

        # Load data
        self.data_loader = PhishingDataLoader(data_path, random_seed=random_seed)
        self.all_texts, self.all_labels = self.data_loader.load_data()

        # Partition across banks
        self.partitioner = BankDataPartitioner(
            num_banks=num_banks,
            partition_strategy=partition_strategy,
            random_seed=random_seed,
        )
        self.bank_data = self.partitioner.partition(self.all_texts, self.all_labels)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logger.info(
            f"Initialized {num_banks} banks with {partition_strategy} partitioning"
        )

    def get_bank_data(
        self,
        bank_id: int,
        create_val_split: bool = True,
    ) -> Dict[str, Any]:
        """
        Get data for a specific bank.

        Args:
            bank_id: Bank ID
            create_val_split: Whether to create train/val split

        Returns:
            Dictionary with train/val texts, labels, and dataloaders
        """
        if bank_id not in self.bank_data:
            raise ValueError(f"Bank {bank_id} does not exist")

        texts, labels = self.bank_data[bank_id]

        if create_val_split:
            train_texts, train_labels, val_texts, val_labels = (
                self.data_loader.create_train_val_split(texts, labels, self.val_size)
            )

            train_loader = self.data_loader.create_dataloaders(
                train_texts, train_labels, self.tokenizer
            )
            val_loader = self.data_loader.create_dataloaders(
                val_texts, val_labels, self.tokenizer, shuffle=False
            )

            return {
                "train_texts": train_texts,
                "train_labels": train_labels,
                "val_texts": val_texts,
                "val_labels": val_labels,
                "train_loader": train_loader,
                "val_loader": val_loader,
            }
        else:
            loader = self.data_loader.create_dataloaders(
                texts, labels, self.tokenizer
            )
            return {
                "texts": texts,
                "labels": labels,
                "loader": loader,
            }

    def get_all_banks_data(self) -> Dict[int, Dict[str, Any]]:
        """Get data for all banks."""
        bank_data = {}
        for bank_id in range(self.num_banks):
            bank_data[bank_id] = self.get_bank_data(bank_id)
        return bank_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data distribution."""
        stats = {
            "num_banks": self.num_banks,
            "total_samples": len(self.all_texts),
            "partition_strategy": self.partition_strategy,
            "banks": {},
        }

        for bank_id, (texts, labels) in self.bank_data.items():
            stats["banks"][bank_id] = {
                "num_samples": len(texts),
                "label_distribution": {
                    "legitimate": labels.count(0),
                    "phishing": labels.count(1),
                },
                "phishing_ratio": labels.count(1) / len(labels) if labels else 0,
            }

        return stats
