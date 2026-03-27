"""Tests for game theory utility functions."""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.game_theory.utility import AttackerUtility, DefenderUtility


@pytest.fixture
def attacker_utility():
    """Create attacker utility instance."""
    return AttackerUtility()


@pytest.fixture
def defender_utility():
    """Create defender utility instance."""
    return DefenderUtility()


def test_attacker_utility_computation(attacker_utility):
    """Test attacker utility computation."""
    # Successful attack with low cost
    utility = attacker_utility.compute(
        attack_success=True,
        evasion_success=True,
        cost=1.0,
        model_accuracy=0.7,
        exposure_risk=0.1,
    )

    assert utility > 0  # Should be positive for successful attack


def test_attacker_utility_failure(attacker_utility):
    """Test attacker utility when attack fails."""
    # Failed attack with high cost
    utility = attacker_utility.compute(
        attack_success=False,
        evasion_success=False,
        cost=10.0,
        model_accuracy=0.95,
        exposure_risk=0.5,
    )

    # Should be lower (potentially negative) for failed attack
    failed_utility = utility

    # Compare to successful attack
    success_utility = attacker_utility.compute(
        attack_success=True,
        evasion_success=True,
        cost=1.0,
        model_accuracy=0.7,
        exposure_risk=0.1,
    )

    assert success_utility > failed_utility


def test_defender_utility_computation(defender_utility):
    """Test defender utility computation."""
    # Good defense with high accuracy
    utility = defender_utility.compute(
        detection_rate=0.9,
        false_positive_rate=0.05,
        model_accuracy=0.95,
        cost=2.0,
    )

    assert utility > 0  # Should be positive for good defense


def test_defender_utility_high_fp(defender_utility):
    """Test defender utility with high false positive rate."""
    # High FP rate should reduce utility
    utility = defender_utility.compute(
        detection_rate=0.9,
        false_positive_rate=0.3,  # High FP
        model_accuracy=0.95,
        cost=2.0,
    )

    # Compare to low FP scenario
    low_fp_utility = defender_utility.compute(
        detection_rate=0.9,
        false_positive_rate=0.05,
        model_accuracy=0.95,
        cost=2.0,
    )

    assert low_fp_utility > utility


def test_utility_weights():
    """Test custom utility weights."""
    custom_attacker = AttackerUtility(
        success_weight=2.0,
        cost_weight=0.5,
        exposure_weight=1.0,
    )

    utility = custom_attacker.compute(
        attack_success=True,
        evasion_success=True,
        cost=1.0,
        model_accuracy=0.7,
    )

    assert utility > 0
