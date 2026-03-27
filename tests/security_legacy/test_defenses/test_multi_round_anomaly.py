"""Tests for multi-round anomaly detection defense."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.defenses.multi_round_anomaly import MultiRoundAnomalyDetection
from src.defenses.base import DefenderObservability, DefenseHistory


@pytest.fixture
def defense():
    """Create defense instance."""
    observability = DefenderObservability(sees_gradients=True)
    return MultiRoundAnomalyDetection(
        defender_observability=observability,
        window_size=5,
        threshold_method="static",
        static_threshold=3.0,
    )


@pytest.fixture
def client_updates():
    """Create mock client updates."""
    updates = []
    for i in range(5):
        # Create update with parameters
        params = {
            "layer1.weight": torch.randn(32, 100) * 0.1,
            "layer1.bias": torch.randn(32) * 0.1,
            "layer2.weight": torch.randn(16, 32) * 0.1,
            "layer2.bias": torch.randn(16) * 0.1,
        }
        updates.append((i, params))
    return updates


def test_anomaly_detection_initialization(defense):
    """Test defense initialization."""
    assert defense.window_size == 5
    assert defense.threshold_method == "static"
    assert defense.current_threshold == 3.0


def test_anomaly_detection_execution(defense, client_updates):
    """Test anomaly detection execution."""
    history = DefenseHistory()

    malicious_ids, metadata = defense.detect(client_updates, round_num=1, history=history)

    assert "anomaly_scores" in metadata
    assert "thresholds" in metadata
    assert len(malicious_ids) <= len(client_updates)
    assert isinstance(malicious_ids, list)


def test_anomaly_score_computation(defense, client_updates):
    """Test anomaly score computation."""
    history = DefenseHistory()

    # Add some history
    for i, params in client_updates[:3]:
        history.add_client_behavior(
            i, round_num=i,
            behavior={"update_norm": defense._compute_update_norm(params)}
        )

    # Compute score for a client with history
    score = defense.compute_anomaly_score(0, client_updates[0][1], history)

    assert score >= 0
    assert isinstance(score, float)


def test_anomaly_detection_adaptation(defense):
    """Test defense adaptation."""
    history = DefenseHistory()

    # Add a record with low FP rate
    from src.defenses.base import DefenseRecord
    record = DefenseRecord(
        round_num=1,
        defense_type="multi_round_anomaly",
        detected_malicious=[1],
        false_positives=[],
        false_negatives=[],
        detection_rate=1.0,
        false_positive_rate=0.0,
        cost=1.0,
    )
    history.add_record(record)

    initial_threshold = defense.current_threshold
    defense.adapt_to_attack(attack_detected=True, attack_type="label_flip", history=history)

    # Threshold should decrease (increase sensitivity)
    assert defense.current_threshold < initial_threshold


def test_defense_cost_computation(defense, client_updates):
    """Test cost computation."""
    history = DefenseHistory()
    malicious_ids, metadata = defense.detect(client_updates, round_num=1, history=history)

    cost = defense.compute_cost(metadata)
    assert cost >= 1.0  # Base cost
