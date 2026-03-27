"""Tests for defense-aware label flip attack."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.attacks.label_flip import DefenseAwareLabelFlip
from src.attacks.base import AttackerKnowledge, AttackConfig, AttackHistory
from src.utils.models import MLP


@pytest.fixture
def attack():
    """Create attack instance."""
    knowledge = AttackerKnowledge(knows_defense=True)
    config = AttackConfig(attack_type="label_flip")
    return DefenseAwareLabelFlip(
        attacker_knowledge=knowledge,
        attack_config=config,
        flip_ratio=0.3,
    )


@pytest.fixture
def model_and_data():
    """Create model and data."""
    model = MLP(input_dim=100, hidden_dims=[32, 16], num_classes=2)
    X = torch.randn(100, 100)
    y = torch.randint(0, 2, (100,))
    return model, X, y


def test_label_flip_initialization(attack):
    """Test attack initialization."""
    assert attack.flip_ratio == 0.3
    assert attack.current_flip_ratio == 0.3
    assert attack.evasion_strategy == "stay_under_bound"


def test_label_flip_execution(attack, model_and_data):
    """Test label flip execution."""
    model, X, y = model_and_data
    history = AttackHistory()

    poisoned_model, metadata = attack.execute(model, (X, y), round_num=1, history=history)

    assert "num_flips" in metadata
    assert metadata["num_flips"] > 0
    assert metadata["num_flips"] <= len(y)


def test_label_flip_adaptation(attack):
    """Test attack adaptation."""
    history = AttackHistory()

    # When detected, should reduce flip ratio
    initial_ratio = attack.current_flip_ratio
    attack.adapt_to_defense(defense_detected=True, defense_type="anomaly", history=history)
    assert attack.current_flip_ratio < initial_ratio

    # When not detected, should increase flip ratio
    attack.adapt_to_defense(defense_detected=False, defense_type=None, history=history)
    # May not increase beyond initial ratio
    assert attack.current_flip_ratio <= attack.flip_ratio


def test_label_flip_cost_computation(attack, model_and_data):
    """Test cost computation."""
    model, X, y = model_and_data
    history = AttackHistory()

    poisoned_model, metadata = attack.execute(model, (X, y), round_num=1, history=history)
    cost = attack.compute_cost(metadata)

    assert cost > 0
    assert isinstance(cost, float)
