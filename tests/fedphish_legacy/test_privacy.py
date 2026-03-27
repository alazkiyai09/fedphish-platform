"""Unit tests for privacy modules."""

import pytest
import numpy as np

from fedphish.privacy.dp import DifferentialPrivacy, GradientClipper, PrivacyAccountant
from fedphish.privacy.ht2ml import PrivacyLevel, HT2MLAggregator


class TestGradientClipper:
    """Test GradientClipper."""

    @pytest.fixture
    def sample_gradients(self):
        """Create sample gradients."""
        return np.random.randn(10, 100) * 2  # Norm > 1

    def test_flat_clip(self, sample_gradients):
        """Test flat clipping."""
        clipper = GradientClipper(clipping_norm=1.0, clipping_method="flat")

        for grad in sample_gradients:
            clipped, clip_factor, norm = clipper.clip_gradients(grad)

            # Check clipping happened
            assert clip_factor <= 1.0
            assert np.linalg.norm(clipped.flatten()) <= 1.0 + 1e-6

    def test_adaptive_clip(self, sample_gradients):
        """Test adaptive clipping."""
        clipper = GradientClipper(clipping_norm=1.0, clipping_method="adaptive")

        # Clip multiple times to build history
        for grad in sample_gradients[:5]:
            clipper.clip_gradients(grad)

        # Now test with new gradient
        clipped, clip_factor, norm = clipper.clip_gradients(sample_gradients[0])

        assert clipped is not None


class TestDifferentialPrivacy:
    """Test DifferentialPrivacy."""

    def test_add_noise(self):
        """Test adding DP noise."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            clipping_norm=1.0,
        )

        gradients = np.random.randn(100) * 0.5
        noisy = dp.add_noise(gradients)

        # Check shape preserved
        assert noisy.shape == gradients.shape

        # Check noise was added (not exactly the same)
        assert not np.allclose(noisy, gradients)

    def test_compute_privacy_spent(self):
        """Test computing privacy spend."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        epsilon, delta = dp.compute_privacy_spent(
            num_steps=100,
            sampling_probability=0.1,
        )

        assert epsilon > 0
        assert delta == 1e-5


class TestPrivacyAccountant:
    """Test PrivacyAccountant."""

    def test_initialization(self):
        """Test accountant initialization."""
        accountant = PrivacyAccountant(target_epsilon=10.0)

        assert accountant.target_epsilon == 10.0

    def test_record_step(self):
        """Test recording privacy step."""
        accountant = PrivacyAccountant()

        accountant.record_step(
            noise_multiplier=1.0,
            sampling_probability=0.1,
            num_steps=100,
        )

        assert len(accountant.history) == 1

    def test_get_privacy_spent(self):
        """Test getting total privacy spent."""
        accountant = PrivacyAccountant()

        accountant.record_step(
            noise_multiplier=1.0,
            sampling_probability=0.1,
            num_steps=100,
        )

        epsilon, delta = accountant.get_privacy_spent()

        assert epsilon > 0
        assert delta == 1e-5


class TestHT2MLAggregator:
    """Test HT2ML Aggregator."""

    @pytest.fixture
    def sample_updates(self):
        """Create sample updates."""
        return [np.random.randn(100) * 0.1 for _ in range(5)]

    def test_aggregate_level1(self, sample_updates):
        """Test aggregation at privacy level 1."""
        aggregator = HT2MLAggregator(
            privacy_level=PrivacyLevel.LEVEL_1,
        )

        result = aggregator.aggregate(
            updates=sample_updates,
            operation="average",
        )

        assert result.shape == sample_updates[0].shape
        # Level 1 should just average
        expected = np.mean(sample_updates, axis=0)
        assert np.allclose(result, expected)

    def test_aggregate_level3(self, sample_updates):
        """Test aggregation at privacy level 3."""
        aggregator = HT2MLAggregator(
            privacy_level=PrivacyLevel.LEVEL_3,
            use_real_tee=False,
        )

        result = aggregator.aggregate(
            updates=sample_updates,
            operation="average",
            encrypted_updates=None,  # No HE in test
        )

        assert result.shape == sample_updates[0].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
