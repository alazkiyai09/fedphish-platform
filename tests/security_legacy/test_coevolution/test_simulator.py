"""Tests for co-evolution simulator."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.coevolution.simulator import CoevolutionSimulator, CoevolutionConfig
from src.coevolution.analyzer import CoevolutionAnalyzer
from src.attacks.label_flip import DefenseAwareLabelFlip
from src.attacks.base import AttackerKnowledge, AttackConfig
from src.defenses.multi_round_anomaly import MultiRoundAnomalyDetection
from src.defenses.base import DefenderObservability


@pytest.fixture
def coevolution_config():
    """Create co-evolution config."""
    return CoevolutionConfig(
        num_rounds=5,
        num_clients=5,
        num_malicious=1,
        num_samples=1000,
        num_features=50,
    )


@pytest.fixture
def attack():
    """Create attack instance."""
    knowledge = AttackerKnowledge(knows_defense=True)
    config = AttackConfig(attack_type="label_flip")
    return DefenseAwareLabelFlip(
        attacker_knowledge=knowledge,
        attack_config=config,
        flip_ratio=0.2,
    )


@pytest.fixture
def defense():
    """Create defense instance."""
    observability = DefenderObservability()
    return MultiRoundAnomalyDetection(
        defender_observability=observability,
        window_size=3,
        threshold_method="static",
        static_threshold=5.0,
    )


def test_simulator_initialization(coevolution_config, attack, defense):
    """Test simulator initialization."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)

    assert simulator.config == coevolution_config
    assert simulator.attack == attack
    assert simulator.defense == defense
    assert simulator.server is not None
    assert len(simulator.clients) == coevolution_config.num_clients
    assert len(simulator.malicious_client_ids) == coevolution_config.num_malicious


def test_simulator_run(coevolution_config, attack, defense):
    """Test simulator execution."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    assert result.history is not None
    assert len(result.history.rounds) > 0
    assert result.total_time > 0
    assert "attack_type" in result.metadata
    assert "defense_type" in result.metadata


def test_analyzer_initialization(coevolution_config, attack, defense):
    """Test analyzer initialization."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    analyzer = CoevolutionAnalyzer(result)

    assert analyzer.history == result.history


def test_analyzer_final_metrics(coevolution_config, attack, defense):
    """Test final metrics computation."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    analyzer = CoevolutionAnalyzer(result)
    metrics = analyzer.compute_final_metrics()

    assert "final_attack_success_rate" in metrics
    assert "final_detection_rate" in metrics
    assert "final_model_accuracy" in metrics
    assert "total_attacker_cost" in metrics
    assert "total_defender_cost" in metrics


def test_analyzer_trends(coevolution_config, attack, defense):
    """Test trend analysis."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    analyzer = CoevolutionAnalyzer(result)
    trends = analyzer.analyze_trends()

    assert "attack_success_rate_trend" in trends
    assert "detection_rate_trend" in trends
    assert "model_accuracy_trend" in trends


def test_analyzer_arms_race(coevolution_config, attack, defense):
    """Test arms race detection."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    analyzer = CoevolutionAnalyzer(result)
    arms_race = analyzer.identify_arms_race()

    assert "arms_race" in arms_race
    assert isinstance(arms_race["arms_race"], bool)


def test_analyzer_equilibrium(coevolution_config, attack, defense):
    """Test equilibrium detection."""
    simulator = CoevolutionSimulator(coevolution_config, attack, defense)
    result = simulator.run()

    analyzer = CoevolutionAnalyzer(result)
    eq_metrics = analyzer.compute_equilibrium_metrics()

    assert "equilibrium_reached" in eq_metrics
    assert isinstance(eq_metrics["equilibrium_reached"], bool)
