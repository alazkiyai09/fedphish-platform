"""Tests for benchmark correctness."""

import numpy as np
import pytest
from src.metrics.classification import compute_accuracy, compute_auprc


class TestBenchmarkCorrectness:
    """Test benchmark correctness."""

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        accuracy = compute_accuracy(y_true, y_pred)

        assert accuracy == 0.8  # 4 out of 5 correct

    def test_auprc_calculation(self):
        """Test AUPRC calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_scores = np.array([0.2, 0.8, 0.6, 0.3, 0.9])

        auprc = compute_auprc(y_true, y_scores)

        assert 0 <= auprc <= 1

    def test_reproducibility_same_seed(self):
        """Test that same seed produces same results."""
        from src.utils.reproducibility import set_seed

        set_seed(42)
        data1 = np.random.randn(10)

        set_seed(42)
        data2 = np.random.randn(10)

        assert np.allclose(data1, data2)

    def test_reproducibility_different_seeds(self):
        """Test that different seeds produce different results."""
        from src.utils.reproducibility import set_seed

        set_seed(42)
        data1 = np.random.randn(10)

        set_seed(43)
        data2 = np.random.randn(10)

        assert not np.allclose(data1, data2)
