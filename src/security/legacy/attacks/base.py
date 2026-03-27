"""Base attack class and related data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AttackerKnowledge:
    """What the attacker knows about the system."""

    knows_aggregation: bool = True  # Knows which aggregation method is used
    knows_defense: bool = True  # Knows which defense mechanism is used
    knows_thresholds: bool = False  # Knows exact threshold values
    knows_data_distribution: bool = True  # Knows data distribution
    knows_other_clients: bool = False  # Knows about other honest clients
    knows_num_malicious: bool = True  # Knows number of malicious clients

    # Specific knowledge
    aggregation_method: Optional[str] = None
    defense_type: Optional[str] = None
    norm_bound: Optional[float] = None


@dataclass
class AttackConfig:
    """Attack configuration."""

    attack_type: str = "label_flip"
    strength: float = 1.0
    budget: float = 1000.0
    max_cost: float = 1000.0

    # Attack-specific parameters
    flip_ratio: float = 0.3
    injection_rate: float = 0.1
    poison_strength: float = 5.0

    # Targeting
    target_class: Optional[int] = None
    target_phishing_type: Optional[str] = None

    # Adaptation
    can_adapt: bool = True
    adaptation_frequency: int = 1  # Adapt every N rounds


@dataclass
class AttackRecord:
    """Record of a single attack execution."""

    round_num: int
    attack_type: str
    success: bool
    detected: bool
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackHistory:
    """History of attacks for adaptation."""

    records: List[AttackRecord] = field(default_factory=list)

    def add_record(self, record: AttackRecord) -> None:
        """Add attack record to history."""
        self.records.append(record)

    def get_recent_records(self, num_rounds: int = 5) -> List[AttackRecord]:
        """Get recent attack records."""
        return self.records[-num_rounds:]

    def get_detection_rate(self, num_rounds: int = 10) -> float:
        """Get detection rate over recent rounds."""
        recent = self.get_recent_records(num_rounds)
        if not recent:
            return 0.0
        detected_count = sum(1 for r in recent if r.detected)
        return detected_count / len(recent)

    def get_success_rate(self, num_rounds: int = 10) -> float:
        """Get success rate over recent rounds."""
        recent = self.get_recent_records(num_rounds)
        if not recent:
            return 0.0
        success_count = sum(1 for r in recent if r.success)
        return success_count / len(recent)

    def get_total_cost(self) -> float:
        """Get total attack cost."""
        return sum(r.cost for r in self.records)


class BaseAttack(ABC):
    """Base class for all attacks."""

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledge,
        attack_config: AttackConfig,
    ):
        """
        Initialize attack.

        Args:
            attacker_knowledge: What attacker knows
            attack_config: Attack configuration
        """
        self.knowledge = attacker_knowledge
        self.config = attack_config

    @abstractmethod
    def execute(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        round_num: int,
        history: AttackHistory,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute attack and return poisoned model + metadata.

        Args:
            model: Model to attack
            data: Training data (X, y)
            round_num: Current round number
            history: Attack history for adaptation

        Returns:
            (poisoned_model, attack_metadata)
        """
        pass

    @abstractmethod
    def adapt_to_defense(
        self,
        defense_detected: bool,
        defense_type: Optional[str],
        history: AttackHistory,
    ) -> None:
        """
        Adapt attack parameters based on defense feedback.

        Args:
            defense_detected: Whether defense was triggered
            defense_type: Type of defense that detected the attack
            history: Attack history
        """
        pass

    def compute_cost(self, metadata: Dict[str, Any]) -> float:
        """
        Compute attack cost based on metadata.

        Args:
            metadata: Attack execution metadata

        Returns:
            Computed cost
        """
        # Base cost: computational cost
        base_cost = 1.0

        # Add cost based on attack strength
        strength_cost = metadata.get("strength", 1.0) * 0.1

        # Add cost for evasion techniques
        evasion_cost = 0.0
        if metadata.get("used_evasion", False):
            evasion_cost = 2.0

        return base_cost + strength_cost + evasion_cost

    def get_attack_type(self) -> str:
        """Get attack type name."""
        return self.config.attack_type
