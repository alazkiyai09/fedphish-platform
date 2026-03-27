"""Co-evolution simulator."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .round import CoevolutionRound, RoundResult
from .history import CoevolutionHistory, RoundMetrics
from ..attacks.base import BaseAttack, AttackHistory, AttackConfig, AttackerKnowledge
from ..defenses.base import BaseDefense, DefenseHistory, DefenseConfig, DefenderObservability
from ..fl.client import FLClient
from ..fl.server import FLServer
from ..utils import load_phishing_data, partition_data, create_model

logger = logging.getLogger(__name__)


@dataclass
class CoevolutionConfig:
    """Co-evolution configuration."""

    num_rounds: int = 20
    num_clients: int = 10
    num_malicious: int = 2
    equilibrium_window: int = 5
    equilibrium_threshold: float = 0.01

    # Data
    num_samples: int = 10000
    num_features: int = 100
    num_classes: int = 2

    # FL
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 5
    aggregation: str = "fedavg"

    # Model
    model_type: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class CoevolutionResult:
    """Result of co-evolution simulation."""

    history: CoevolutionHistory
    equilibrium_reached: bool
    equilibrium_round: int = -1
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoevolutionSimulator:
    """Main co-evolution simulator."""

    def __init__(
        self,
        config: CoevolutionConfig,
        attack: BaseAttack,
        defense: BaseDefense,
    ):
        """
        Initialize co-evolution simulator.

        Args:
            config: Co-evolution configuration
            attack: Attack to use
            defense: Defense to use
        """
        self.config = config
        self.attack = attack
        self.defense = defense

        # Initialize components
        self._setup_federated_learning()

    def _setup_federated_learning(self) -> None:
        """Set up federated learning components."""
        logger.info("Setting up federated learning...")

        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_phishing_data(
            num_samples=self.config.num_samples,
            num_features=self.config.num_features,
            num_classes=self.config.num_classes,
        )

        # Partition data
        train_partitions = partition_data(
            X_train, y_train,
            num_clients=self.config.num_clients,
            distribution="iid",
        )

        val_partitions = partition_data(
            X_val, y_val,
            num_clients=self.config.num_clients,
            distribution="iid",
        )

        # Create global model
        global_model = create_model(
            model_type=self.config.model_type,
            input_dim=self.config.num_features,
            num_classes=self.config.num_classes,
            hidden_dims=self.config.hidden_dims,
        )

        # Create server
        self.server = FLServer(
            model=global_model,
            num_clients=self.config.num_clients,
            learning_rate=self.config.learning_rate,
            aggregation_strategy=self.config.aggregation,
            defense=None,  # Defense applied separately
        )

        # Create clients
        self.clients = []
        for client_id in range(self.config.num_clients):
            model = create_model(
                model_type=self.config.model_type,
                input_dim=self.config.num_features,
                num_classes=self.config.num_classes,
                hidden_dims=self.config.hidden_dims,
            )

            client = FLClient(
                client_id=client_id,
                model=model,
                train_data=train_partitions[client_id],
                val_data=val_partitions[client_id],
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                local_epochs=self.config.local_epochs,
            )

            self.clients.append(client)

        # Assign malicious clients
        self.malicious_client_ids = list(range(self.config.num_malicious))

        logger.info(
            f"Setup complete: {len(self.clients)} clients, "
            f"{len(self.malicious_client_ids)} malicious"
        )

        # Store test data
        self.test_data = (X_test, y_test)

    def run(self) -> CoevolutionResult:
        """
        Run full co-evolution simulation.

        Returns:
            CoevolutionResult
        """
        import time
        start_time = time.time()

        logger.info(f"Starting co-evolution simulation ({self.config.num_rounds} rounds)")

        history = CoevolutionHistory()
        attack_history = AttackHistory()
        defense_history = DefenseHistory()

        equilibrium_reached = False
        equilibrium_round = -1

        # Create round executor
        round_executor = CoevolutionRound(
            attack=self.attack,
            defense=self.defense,
            server=self.server,
            clients=self.clients,
            malicious_client_ids=self.malicious_client_ids,
            test_data=self.test_data,
        )

        for round_num in range(1, self.config.num_rounds + 1):
            # Execute round
            result = round_executor.execute(
                round_num, attack_history, defense_history
            )

            # Record metrics
            metrics = RoundMetrics(
                round_num=round_num,
                attack_success_rate=result.attack_success_rate,
                detection_rate=result.detection_rate,
                false_positive_rate=result.false_positive_rate,
                model_accuracy=result.model_accuracy,
                defense_overhead=result.defense_overhead,
                attacker_cost=result.attacker_cost,
                defender_cost=result.defender_cost,
                attack_type=result.attack_type,
                defense_type=result.defense_type,
            )
            history.add_round(metrics)

            # Check equilibrium
            if history.check_equilibrium(
                self.config.equilibrium_window,
                self.config.equilibrium_threshold
            ):
                equilibrium_reached = True
                equilibrium_round = round_num
                logger.info(f"Equilibrium reached at round {round_num}")
                break

        total_time = time.time() - start_time

        result = CoevolutionResult(
            history=history,
            equilibrium_reached=equilibrium_reached,
            equilibrium_round=equilibrium_round,
            total_time=total_time,
            metadata={
                "num_rounds": len(history.rounds),
                "attack_type": self.attack.get_attack_type(),
                "defense_type": self.defense.get_defense_type(),
            },
        )

        logger.info(
            f"Co-evolution simulation complete: "
            f"{len(history.rounds)} rounds, {total_time:.2f}s, "
            f"equilibrium={'yes' if equilibrium_reached else 'no'}"
        )

        return result
