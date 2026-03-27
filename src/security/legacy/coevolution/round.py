"""Single co-evolution round execution."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ..attacks.base import BaseAttack, AttackHistory
from ..defenses.base import BaseDefense, DefenseHistory
from ..fl.client import FLClient
from ..fl.server import FLServer

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single co-evolution round."""

    round_num: int
    attack_success: bool
    attack_detected: bool
    detection_rate: float
    false_positive_rate: float
    model_accuracy: float
    attack_success_rate: float
    defense_overhead: float
    attacker_cost: float
    defender_cost: float
    attack_type: str
    defense_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoevolutionRound:
    """Execute a single co-evolution round."""

    def __init__(
        self,
        attack: BaseAttack,
        defense: BaseDefense,
        server: FLServer,
        clients: List[FLClient],
        malicious_client_ids: List[int],
        test_data: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Initialize co-evolution round.

        Args:
            attack: Attack to use
            defense: Defense to use
            server: FL server
            clients: List of all clients
            malicious_client_ids: List of malicious client IDs
            test_data: Test data for evaluation
        """
        self.attack = attack
        self.defense = defense
        self.server = server
        self.clients = clients
        self.malicious_client_ids = malicious_client_ids
        self.test_data = test_data

    def execute(
        self,
        round_num: int,
        attack_history: AttackHistory,
        defense_history: DefenseHistory,
    ) -> RoundResult:
        """
        Execute single co-evolution round.

        Args:
            round_num: Round number
            attack_history: Attack history
            defense_history: Defense history

        Returns:
            RoundResult
        """
        logger.info(f"Executing co-evolution round {round_num}")
        start_time = time.time()

        # Phase 1: Attacker adapts (if round > 1)
        if round_num > 1 and len(attack_history.records) > 0:
            last_detected = attack_history.records[-1].detected
            self.attack.adapt_to_defense(
                last_detected,
                self.defense.get_defense_type(),
                attack_history,
            )

        # Phase 2: Defense adapts (if round > 1)
        if round_num > 1 and len(defense_history.records) > 0:
            last_detected = defense_history.records[-1].detected_malicious != []
            self.defense.adapt_to_attack(
                last_detected,
                self.attack.get_attack_type(),
                defense_history,
            )

        # Phase 3: Federated learning with attack
        client_updates = []
        attacker_cost = 0.0

        for client in self.clients:
            # Check if client is malicious
            is_malicious = client.client_id in self.malicious_client_ids

            if is_malicious:
                # Apply attack
                train_data = next(iter(client.train_loader))
                X_batch, y_batch = train_data

                poisoned_model, attack_metadata = self.attack.execute(
                    client.model,
                    (X_batch, y_batch),
                    round_num,
                    attack_history,
                )

                # Get parameters from poisoned model
                poisoned_params = {
                    name: param.data
                    for name, param in poisoned_model.named_parameters()
                }

                client_updates.append((client.client_id, poisoned_params))
                attacker_cost += self.attack.compute_cost(attack_metadata)

            else:
                # Normal training
                params, _ = client.train()
                client_updates.append((client.client_id, params))

        # Phase 4: Apply defense
        defense_start_time = time.time()
        detected_malicious, detection_metadata = self.defense.detect(
            client_updates, round_num, defense_history
        )
        defense_time = time.time() - defense_start_time

        # Remove detected clients
        filtered_updates = [
            (cid, params) for cid, params in client_updates
            if cid not in detected_malicious
        ]

        # Phase 5: Aggregate updates
        aggregated_params = self.server.aggregate(filtered_updates)
        self.server.update_model(aggregated_params)

        # Update all clients with new global model
        for client in self.clients:
            client.update_parameters(self.server.get_parameters())

        # Phase 6: Evaluate
        test_metrics = self.server.evaluate(self.test_data)
        model_accuracy = test_metrics["accuracy"] / 100.0

        # Compute metrics
        detection_rate = len(detected_malicious) / len(self.malicious_client_ids)
        false_positives = [
            cid for cid in detected_malicious
            if cid not in self.malicious_client_ids
        ]
        false_positive_rate = len(false_positives) / (
            len(self.clients) - len(self.malicious_client_ids)
        )

        # Attack success: not detected OR detected but model still degraded
        attack_success = (
            len(detected_malicious) < len(self.malicious_client_ids) or
            model_accuracy < 0.9
        )

        attack_success_rate = 1.0 - detection_rate

        defense_overhead = defense_time
        defender_cost = self.defense.compute_cost(detection_metadata)

        total_time = time.time() - start_time

        # Record histories
        from ..attacks.base import AttackRecord
        from ..defenses.base import DefenseRecord

        attack_record = AttackRecord(
            round_num=round_num,
            attack_type=self.attack.get_attack_type(),
            success=attack_success,
            detected=len(detected_malicious) > 0,
            cost=attacker_cost,
        )
        attack_history.add_record(attack_record)

        defense_record = DefenseRecord(
            round_num=round_num,
            defense_type=self.defense.get_defense_type(),
            detected_malicious=detected_malicious,
            false_positives=false_positives,
            false_negatives=[
                cid for cid in self.malicious_client_ids
                if cid not in detected_malicious
            ],
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            cost=defender_cost,
        )
        defense_history.add_record(defense_record)

        result = RoundResult(
            round_num=round_num,
            attack_success=attack_success,
            attack_detected=len(detected_malicious) > 0,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            model_accuracy=model_accuracy,
            attack_success_rate=attack_success_rate,
            defense_overhead=defense_overhead,
            attacker_cost=attacker_cost,
            defender_cost=defender_cost,
            attack_type=self.attack.get_attack_type(),
            defense_type=self.defense.get_defense_type(),
            metadata={
                "round_time": total_time,
                "num_detected": len(detected_malicious),
            },
        )

        logger.info(
            f"Round {round_num} completed: "
            f"ASR={attack_success_rate:.2%}, DR={detection_rate:.2%}, "
            f"Acc={model_accuracy:.2%}"
        )

        return result
