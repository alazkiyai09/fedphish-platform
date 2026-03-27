"""Unit tests for FedPhish utility modules."""

import pytest
import numpy as np

from fedphish.utils.data import BankDataPartitioner
from fedphish.utils.metrics import DetectionMetrics, PrivacyMetrics
from fedphish.utils.metrics import FederatedMetrics, SecurityMetrics


class TestBankDataPartitioner:
    """Test BankDataPartitioner."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        texts = [
            "Urgent: Verify your account now",
            "Your monthly statement is ready",
            "Free gift click here",
            "Meeting tomorrow at 10am",
        ] * 25  # 100 samples
        labels = [1, 0, 1, 0] * 25  # 50% phishing
        return texts, labels

    def test_partition_iid(self, sample_data):
        """Test IID partitioning."""
        texts, labels = sample_data
        partitioner = BankDataPartitioner(
            num_banks=5,
            partition_strategy="iid",
            random_seed=42,
        )

        bank_data = partitioner.partition(texts, labels)

        assert len(bank_data) == 5
        for bank_id, (bank_texts, bank_labels) in bank_data.items():
            assert len(bank_texts) > 0
            assert len(bank_texts) == len(bank_labels)
            print(f"Bank {bank_id}: {len(bank_texts)} samples")

    def test_partition_non_iid(self, sample_data):
        """Test non-IID partitioning."""
        texts, labels = sample_data
        partitioner = BankDataPartitioner(
            num_banks=5,
            partition_strategy="non-iid",
            alpha=0.5,
            random_seed=42,
        )

        bank_data = partitioner.partition(texts, labels)

        assert len(bank_data) == 5
        # Check for label skew (different from IID)
        phishing_ratios = []
        for bank_id, (bank_texts, bank_labels) in bank_data.items():
            if len(bank_labels) > 0:
                ratio = sum(bank_labels) / len(bank_labels)
                phishing_ratios.append(ratio)

        # Ratios should vary more in non-IID
        assert len(set(phishing_ratios)) > 1

    def test_partition_imbalanced(self, sample_data):
        """Test imbalanced partitioning."""
        texts, labels = sample_data
        partitioner = BankDataPartitioner(
            num_banks=5,
            partition_strategy="imbalanced",
            imbalance_factor=0.3,
            random_seed=42,
        )

        bank_data = partitioner.partition(texts, labels)

        assert len(bank_data) == 5
        # Check for quantity imbalance
        sizes = [len(bank_texts) for bank_texts, _ in bank_data.values()]
        assert max(sizes) > min(sizes) * 2  # At least 2x difference


class TestFederatedMetrics:
    """Test FederatedMetrics."""

    def test_log_round(self):
        """Test logging round metrics."""
        metrics = FederatedMetrics()

        metrics.log_round(
            round_num=1,
            num_clients=5,
            accuracy=0.95,
            loss=0.1,
            communication_cost=1e6,
            convergence_rate=0.01,
            training_time=100.0,
        )

        assert len(metrics.round_metrics) == 1

    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics = FederatedMetrics()

        # Log multiple rounds
        for i in range(3):
            metrics.log_round(
                round_num=i + 1,
                num_clients=5,
                accuracy=0.9 + i * 0.02,
                loss=0.2 - i * 0.05,
                communication_cost=1e6,
                convergence_rate=0.01,
                training_time=100.0,
            )

        summary = metrics.get_summary()

        assert summary["total_rounds"] == 3
        assert summary["final_accuracy"] == 0.94
        assert summary["best_accuracy"] == 0.94


class TestPrivacyMetrics:
    """Test PrivacyMetrics."""

    def test_compute_epsilon_delta(self):
        """Test computing epsilon and delta."""
        metrics = PrivacyMetrics()

        epsilon, delta = metrics.compute_epsilon_delta(
            noise_multiplier=1.0,
            sampling_probability=0.1,
            num_steps=100,
        )

        assert epsilon > 0
        assert delta == 1e-5

    def test_log_privacy_spend(self):
        """Test logging privacy spend."""
        metrics = PrivacyMetrics()

        metrics.log_privacy_spend(
            round_num=1,
            epsilon=0.5,
            delta=1e-5,
            noise_multiplier=1.0,
            clipping_norm=1.0,
        )

        assert len(metrics.privacy_ledger) == 1

    def test_get_total_privacy_cost(self):
        """Test getting total privacy cost."""
        metrics = PrivacyMetrics()

        # Log multiple rounds
        for i in range(3):
            metrics.log_privacy_spend(
                round_num=i + 1,
                epsilon=0.5,
                delta=1e-5,
                noise_multiplier=1.0,
                clipping_norm=1.0,
            )

        total = metrics.get_total_privacy_cost()

        assert total["epsilon"] == 1.5  # 3 * 0.5
        assert total["delta"] == 3e-5


class TestSecurityMetrics:
    """Test SecurityMetrics."""

    def test_log_proof_verification(self):
        """Test logging proof verification."""
        metrics = SecurityMetrics()

        metrics.log_proof_verification(
            round_num=1,
            verification_time=1.0,
            num_proofs=5,
            num_valid=5,
        )

        assert len(metrics.proof_verification_times) == 1

    def test_log_attack_detection(self):
        """Test logging attack detection."""
        metrics = SecurityMetrics()

        metrics.log_attack_detection(
            round_num=1,
            num_malicious=1,
            num_detected=1,
            num_false_positives=0,
            defense_strategy="foolsgold",
        )

        assert len(metrics.attack_detections) == 1

        detection = metrics.attack_detections[0]
        assert detection["precision"] == 1.0
        assert detection["recall"] == 1.0


class TestDetectionMetrics:
    """Test DetectionMetrics."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_prob = np.array([
            [0.9, 0.1],  # Correct
            [0.8, 0.2],  # Correct
            [0.1, 0.9],  # Correct
            [0.6, 0.4],  # Wrong (should be 1)
            [0.7, 0.3],  # Correct
            [0.2, 0.8],  # Correct
        ])
        return y_true, y_pred, y_prob

    def test_compute_metrics(self, sample_predictions):
        """Test computing detection metrics."""
        y_true, y_pred, y_prob = sample_predictions

        metrics = DetectionMetrics.compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 5/6
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_compute_roc_curve(self, sample_predictions):
        """Test computing ROC curve."""
        y_true, y_pred, y_prob = sample_predictions

        fpr, tpr, thresholds = DetectionMetrics.compute_roc_curve(y_true, y_prob)

        assert len(fpr) == len(tpr)
        assert len(fpr) == len(thresholds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
