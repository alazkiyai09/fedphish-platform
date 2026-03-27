"""
Feature extraction for phishing detection.

Extracts engineered features from URLs and emails.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class URLFeatureExtractor:
    """
    Extract lexical and host-based features from URLs.

    Features include:
    - Lexical: length, dot count, special characters, etc.
    - Host-based: IP address, subdomain count, suspicious words
    """

    def __init__(self):
        """Initialize URL feature extractor."""
        # Suspicious words often found in phishing URLs
        self.suspicious_words = {
            "login", "signin", "verify", "account", "secure", "update",
            "confirm", "banking", "wallet", "password", "credential",
            "free", "gift", "winner", "urgent", "immediate", "alert",
        }

        # Suspicious TLDs
        self.suspicious_tlds = {
            ".xyz", ".top", ".zip", ".tk", ".ml", ".ga", ".cf",
            ".gq", ".cc", ".pw", ".biz",
        }

    def extract(
        self,
        url: str,
    ) -> Dict[str, float]:
        """
        Extract features from a URL.

        Args:
            url: URL string

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        try:
            parsed = urlparse(url)

            # Lexical features
            features.update(self._extract_lexical_features(url, parsed))

            # Host-based features
            features.update(self._extract_host_features(url, parsed))

            # Path and query features
            features.update(self._extract_path_features(url, parsed))

        except Exception as e:
            logger.warning(f"Error extracting features from URL: {e}")
            # Return default features
            features = {f"default_{i}": 0.0 for i in range(20)}

        return features

    def _extract_lexical_features(
        self,
        url: str,
        parsed,
    ) -> Dict[str, float]:
        """Extract lexical features."""
        features = {}

        # Length features
        features["url_length"] = len(url)
        features["hostname_length"] = len(parsed.hostname) if parsed.hostname else 0
        features["path_length"] = len(parsed.path) if parsed.path else 0

        # Character counts
        features["num_dots"] = url.count(".")
        features["num_hyphens"] = url.count("-")
        features["num_underscores"] = url.count("_")
        features["num_slashes"] = url.count("/")
        features["num_at"] = url.count("@")
        features["num_equals"] = url.count("=")
        features["num_question_marks"] = url.count("?")
        features["num_ampersands"] = url.count("&")
        features["num_percent"] = url.count("%")

        # Special character ratio
        special_chars = sum([
            features["num_dots"], features["num_hyphens"],
            features["num_underscores"], features["num_at"],
            features["num_percent"],
        ])
        features["special_char_ratio"] = special_chars / len(url) if url else 0

        # Digit ratio
        digits = sum(c.isdigit() for c in url)
        features["digit_ratio"] = digits / len(url) if url else 0

        # Uppercase ratio
        uppercase = sum(c.isupper() for c in url)
        features["uppercase_ratio"] = uppercase / len(url) if url else 0

        return features

    def _extract_host_features(
        self,
        url: str,
        parsed,
    ) -> Dict[str, float]:
        """Extract host-based features."""
        features = {}

        hostname = parsed.hostname or ""

        # IP address check
        features["has_ip_address"] = float(bool(
            re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname)
        ))

        # Subdomain count
        parts = hostname.split(".")
        features["subdomain_count"] = max(0, len(parts) - 2)

        # Domain length
        if len(parts) >= 2:
            features["domain_length"] = len(parts[-2])
        else:
            features["domain_length"] = 0

        # Suspicious TLD
        tld = "." + parts[-1] if parts else ""
        features["has_suspicious_tld"] = float(tld.lower() in self.suspicious_tlds)

        # Suspicious words
        hostname_lower = hostname.lower()
        suspicious_count = sum(
            1 for word in self.suspicious_words
            if word in hostname_lower
        )
        features["suspicious_word_count"] = suspicious_count

        return features

    def _extract_path_features(
        self,
        url: str,
        parsed,
    ) -> Dict[str, float]:
        """Extract path and query features."""
        features = {}

        path = parsed.path or ""
        query = parsed.query or ""

        # Path depth
        features["path_depth"] = path.count("/")

        # Query parameters
        features["num_query_params"] = query.count("&") + 1 if query else 0

        # Has suspicious words in path/query
        full_path = (path + " " + query).lower()
        suspicious_count = sum(
            1 for word in self.suspicious_words
            if word in full_path
        )
        features["path_suspicious_word_count"] = suspicious_count

        return features

    def extract_batch(
        self,
        urls: List[str],
    ) -> np.ndarray:
        """
        Extract features from multiple URLs.

        Args:
            urls: List of URLs

        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = [self.extract(url) for url in urls]

        # Get all feature names
        all_features = set()
        for f in features_list:
            all_features.update(f.keys())

        # Create feature matrix
        feature_matrix = []
        for features in features_list:
            row = [features.get(f, 0.0) for f in sorted(all_features)]
            feature_matrix.append(row)

        return np.array(feature_matrix)


class EmailFeatureExtractor:
    """
    Extract features from emails for phishing detection.

    Features include:
    - Header features: sender, subject, reply-to
    - Body features: HTML tags, links, urgency words
    - Metadata features: encoding, attachments
    """

    def __init__(self):
        """Initialize email feature extractor."""
        # Urgency words
        self.urgency_words = {
            "urgent", "immediately", "verify", "confirm", "suspended",
            "expire", "limited time", "act now", "warning", "alert",
        }

        # Suspicious HTML tags
        self.suspicious_tags = {
            "<form", "<input", "<iframe", "<script", "<embed",
            "<object", "<link",
        }

    def extract(
        self,
        email_text: str,
    ) -> Dict[str, float]:
        """
        Extract features from email text.

        Args:
            email_text: Email body text

        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        text_lower = email_text.lower()

        # Basic features
        features["email_length"] = len(email_text)
        features["num_words"] = len(email_text.split())

        # Urgency words
        urgency_count = sum(
            1 for word in self.urgency_words
            if word in text_lower
        )
        features["urgency_word_count"] = urgency_count
        features["has_urgency_words"] = float(urgency_count > 0)

        # HTML features
        html_features = self._extract_html_features(email_text)
        features.update(html_features)

        # Link features
        link_features = self._extract_link_features(email_text)
        features.update(link_features)

        # Attachment mentions
        features["mentions_attachment"] = float(
            any(word in text_lower for word in ["attachment", "attached", "file"])
        )

        return features

    def _extract_html_features(
        self,
        email_text: str,
    ) -> Dict[str, float]:
        """Extract HTML-related features."""
        features = {}
        text_lower = email_text.lower()

        # HTML tag counts
        features["num_html_tags"] = email_text.count("<")
        features["has_html"] = float("<html" in text_lower or "<div" in text_lower)

        # Suspicious tags
        suspicious_count = sum(
            email_text.lower().count(tag)
            for tag in self.suspicious_tags
        )
        features["suspicious_tag_count"] = suspicious_count

        # Form tags (very suspicious)
        features["has_form"] = float("<form" in text_lower)
        features["has_input"] = float("<input" in text_lower)

        # JavaScript
        features["has_javascript"] = float(
            "<script" in text_lower or "javascript:" in text_lower
        )

        # CSS obfuscation
        features["has_inline_css"] = float("style=" in email_text)

        return features

    def _extract_link_features(
        self,
        email_text: str,
    ) -> Dict[str, float]:
        """Extract link-related features."""
        features = {}

        # Count links
        http_count = email_text.lower().count("http")
        https_count = email_text.lower().count("https")
        features["num_links"] = http_count + https_count
        features["https_ratio"] = (
            https_count / features["num_links"]
            if features["num_links"] > 0
            else 0
        )

        # IP address links
        ip_links = len(re.findall(
            r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
            email_text
        ))
        features["num_ip_links"] = ip_links

        # URL shorteners
        shorteners = [
            "bit.ly", "tinyurl.com", "goo.gl", "t.co",
            "ow.ly", "is.gd", "buff.ly",
        ]
        shortener_count = sum(
            email_text.lower().count(s)
            for s in shorteners
        )
        features["num_shortener_links"] = shortener_count

        # Link mismatches (display text vs actual URL)
        # Simplified - just count href attributes
        features["num_href"] = email_text.lower().count("href=")

        return features


class TFIDFVectorizer:
    """
    TF-IDF vectorization for text features.

    Converts text to TF-IDF feature vectors.
    """

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency (ratio)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
    ):
        """
        Fit vectorizer on texts.

        Args:
            texts: List of text strings
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Fitted TF-IDF vectorizer with {len(self.vectorizer.vocabulary_)} features")

    def transform(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.

        Args:
            texts: List of text strings

        Returns:
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        return self.vectorizer.transform(texts).toarray()

    def fit_transform(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)


class FeatureFusion:
    """
    Combine different feature types.

    Fuses engineered features, TF-IDF features, and learned embeddings.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize feature fusion.

        Args:
            feature_names: Names of feature types
        """
        self.feature_names = feature_names or [
            "url_features",
            "email_features",
            "tfidf_features",
            "embedding_features",
        ]

    def fuse(
        self,
        feature_dict: Dict[str, np.ndarray],
        method: str = "concatenate",
    ) -> np.ndarray:
        """
        Fuse multiple feature types.

        Args:
            feature_dict: Dictionary of feature name -> array
            method: Fusion method ('concatenate', 'sum', 'average')

        Returns:
            Fused feature vector
        """
        if method == "concatenate":
            return self._concatenate_features(feature_dict)
        elif method == "sum":
            return self._sum_features(feature_dict)
        elif method == "average":
            return self._average_features(feature_dict)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _concatenate_features(
        self,
        feature_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Concatenate features along feature dimension."""
        # Ensure all features have same number of samples
        num_samples = list(feature_dict.values())[0].shape[0]

        fused_list = []
        for name, features in feature_dict.items():
            if features.shape[0] != num_samples:
                raise ValueError(f"Feature {name} has inconsistent samples")

            # Flatten if 2D
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)

            fused_list.append(features)

        return np.concatenate(fused_list, axis=1)

    def _sum_features(
        self,
        feature_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Sum features (requires same dimensionality)."""
        feature_arrays = list(feature_dict.values())

        # Check dimensions match
        ref_shape = feature_arrays[0].shape[1:]
        for arr in feature_arrays[1:]:
            if arr.shape[1:] != ref_shape:
                raise ValueError("Features must have same dimensions for sum fusion")

        return sum(feature_arrays)

    def _average_features(
        self,
        feature_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Average features (requires same dimensionality)."""
        summed = self._sum_features(feature_dict)
        return summed / len(feature_dict)
