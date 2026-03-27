"""Data loading and preprocessing for phishing dataset."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PhishingDataLoader:
    """Load and preprocess phishing URL dataset."""

    def __init__(self, config: DictConfig):
        """
        Initialize data loader.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.raw_path = Path(config.dataset.raw_path)
        self.processed_path = Path(config.dataset.processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Initialize feature extractors
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()

        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the phishing dataset.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Loading data from {self.raw_path}")

        # Load raw data
        if self.raw_path.exists():
            df = pd.read_csv(self.raw_path)
        else:
            # Generate synthetic dataset for testing
            logger.warning(f"Data file not found, generating synthetic dataset")
            df = self._generate_synthetic_data()

        logger.info(f"Loaded {len(df)} samples")

        # Extract features
        X, y = self._extract_features(df)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=self.config.dataset.train_ratio,
            random_state=42,
            stratify=y
        )

        val_size = self.config.dataset.val_ratio / (
            self.config.dataset.val_ratio + self.config.dataset.test_ratio
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_size,
            random_state=42,
            stratify=y_temp
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)

    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataframe.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (features, labels)
        """
        # Assume dataset has 'url' and 'label' columns
        # If not, adjust accordingly

        if 'url' in df.columns:
            urls = df['url'].values
        else:
            urls = df.iloc[:, 0].values  # First column

        if 'label' in df.columns:
            labels = df['label'].values
        elif 'target' in df.columns:
            labels = df['target'].values
        else:
            labels = df.iloc[:, -1].values  # Last column

        # Extract TF-IDF features from URLs
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.features.tfidf_max_features,
                ngram_range=tuple(self.config.features.tfidf_ngram_range),
                lowercase=True,
                preprocessor=self._preprocess_url
            )
            features = self.tfidf_vectorizer.fit_transform(urls)
        else:
            features = self.tfidf_vectorizer.transform(urls)

        return features.toarray(), labels.astype(np.int32)

    def _preprocess_url(self, url: str) -> str:
        """
        Preprocess URL for feature extraction.

        Args:
            url: URL string

        Returns:
            Preprocessed URL
        """
        url = url.lower()
        url = url.replace("https://", "").replace("http://", "")
        url = url.replace("www.", "")
        return url

    def _generate_synthetic_data(
        self,
        num_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Generate synthetic phishing dataset for testing.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Synthetic dataframe
        """
        logger.info(f"Generating {num_samples} synthetic samples")

        # Generate URLs
        legitimate_urls = [
            "https://www.example.com",
            "https://www.bankofamerica.com/login",
            "https://www.chase.com/signin",
            "https://www.paypal.com/us/home",
        ]

        phishing_urls = [
            "http://secure-login.verify-account.com",
            "http://www.paypal-secure.com/login",
            "http://bank-america-verification.net/signin",
            "https://www.appleid-support.site/auth",
        ]

        urls = []
        labels = []

        for _ in range(num_samples // 2):
            # Legitimate
            base = np.random.choice(legitimate_urls)
            urls.append(base)
            labels.append(0)

            # Phishing
            base = np.random.choice(phishing_urls)
            urls.append(base)
            labels.append(1)

        return pd.DataFrame({
            'url': urls,
            'label': labels
        })

    def get_client_data(
        self,
        client_id: int,
        partition_dict: Dict[int, np.ndarray],
        subset: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific client.

        Args:
            client_id: Client ID
            partition_dict: Partition dictionary mapping client_id to indices
            subset: Which subset to use (train, val, test)

        Returns:
            Tuple of (X, y) for the client
        """
        indices = partition_dict[client_id]

        if subset == "train":
            X = self.X_train[indices]
            y = self.y_train[indices]
        elif subset == "val":
            X = self.X_val[indices]
            y = self.y_val[indices]
        else:
            X = self.X_test[indices]
            y = self.y_test[indices]

        return X, y


def load_phishing_dataset(config: DictConfig) -> PhishingDataLoader:
    """
    Convenience function to load phishing dataset.

    Args:
        config: Configuration

    Returns:
        Loaded data loader
    """
    loader = PhishingDataLoader(config)
    loader.load_data()
    return loader
