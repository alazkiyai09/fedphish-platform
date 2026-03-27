"""Base defense class and related data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class DefenderObservability:
    """What the defender can observe."""

    sees_gradients: bool = True  # Can see client gradients
    sees_updates: bool = True  # Can see parameter updates
    sees_data: bool = False  # Cannot see client data (privacy)
    sees_client_ids: bool = True  # Can identify clients
    knows_num_malicious: bool = False  # Doesn't know exact number of attackers
    knows_attacker_type: bool = False  # Doesn't know attack type initially


@dataclass
class DefenseConfig:
    """Defense configuration."""

    defense_type: str = "multi_round_anomaly"
    threshold: float = 2.0
    window_size: int = 10

    # Honeypot-specific
    num_honeypots: int = 3
    honeypot_strategy: str = "random"

    # Forensics-specific
    analysis_method: str = "pca"
    coordination_threshold: float = 0.9

    # Adaptation
    can_adapt: bool = True
    adaptation_frequency: int = 1


@dataclass
class DefenseRecord:
    """Record of a single defense execution."""

    round_num: int
    defense_type: str
    detected_malicious: List[int]
    false_positives: List[int]
    false_negatives: List[int]
    detection_rate: float
    false_positive_rate: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseHistory:
    """History of defense executions for adaptation."""

    records: List[DefenseRecord] = field(default_factory=list)
    client_behaviors: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def add_record(self, record: DefenseRecord) -> None:
        """Add defense record to history."""
        self.records.append(record)

    def add_client_behavior(
        self,
        client_id: int,
        round_num: int,
        behavior: Dict[str, Any],
    ) -> None:
        """Add client behavior for tracking."""
        if client_id not in self.client_behaviors:
            self.client_behaviors[client_id] = []
        self.client_behaviors[client_id].append({
            "round_num": round_num,
            **behavior
        })

    def get_client_history(
        self,
        client_id: int,
        num_rounds: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent client behavior history."""
        if client_id not in self.client_behaviors:
            return []
        return self.client_behaviors[client_id][-num_rounds:]

    def get_avg_detection_rate(self, num_rounds: int = 10) -> float:
        """Get average detection rate over recent rounds."""
        recent = self.records[-num_rounds:]
        if not recent:
            return 0.0
        return sum(r.detection_rate for r in recent) / len(recent)

    def get_avg_fp_rate(self, num_rounds: int = 10) -> float:
        """Get average false positive rate over recent rounds."""
        recent = self.records[-num_rounds:]
        if not recent:
            return 0.0
        return sum(r.false_positive_rate for r in recent) / len(recent)


class BaseDefense(ABC):
    """Base class for all defenses."""

    def __init__(
        self,
        defender_observability: DefenderObservability,
        defense_config: DefenseConfig,
    ):
        """
        Initialize defense.

        Args:
            defender_observability: What defender can observe
            defense_config: Defense configuration
        """
        self.observability = defender_observability
        self.config = defense_config

    @abstractmethod
    def detect(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
        round_num: int,
        history: DefenseHistory,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Detect malicious clients.

        Args:
            client_updates: List of (client_id, parameters) tuples
            round_num: Current round number
            history: Defense history for adaptation

        Returns:
            (malicious_client_ids, detection_metadata)
        """
        pass

    @abstractmethod
    def adapt_to_attack(
        self,
        attack_detected: bool,
        attack_type: Optional[str],
        history: DefenseHistory,
    ) -> None:
        """
        Adapt defense parameters based on attack feedback.

        Args:
            attack_detected: Whether attack was detected
            attack_type: Type of attack detected (if known)
            history: Defense history
        """
        pass

    def compute_cost(self, metadata: Dict[str, Any]) -> float:
        """
        Compute defense cost based on metadata.

        Args:
            metadata: Defense execution metadata

        Returns:
            Computed cost
        """
        # Base cost
        base_cost = 1.0

        # Add cost for complex analysis
        analysis_cost = 0.0
        if metadata.get("used_pca", False):
            analysis_cost += 2.0
        if metadata.get("used_clustering", False):
            analysis_cost += 3.0
        if metadata.get("used_honeypots", False):
            num_honeypots = metadata.get("num_honeypots", 0)
            analysis_cost += num_honeypots * 1.5

        return base_cost + analysis_cost

    def get_defense_type(self) -> str:
        """Get defense type name."""
        return self.config.defense_type
