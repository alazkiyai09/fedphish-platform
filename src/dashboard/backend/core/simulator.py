"""
Federated Learning Simulator for Demo Dashboard.

Simulates FL training with realistic progress, attacks, and defenses.
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from ..config import ScenarioConfig, BankConfig

logger = logging.getLogger(__name__)


class BankState:
    """State of a single bank during training."""

    def __init__(self, config: BankConfig):
        self.config = config
        self.accuracy = 0.5 + random.random() * 0.2  # Start at 0.5-0.7
        self.loss = 1.0 - self.accuracy
        self.status = "idle"  # idle, training, complete, malicious
        self.samples = random.randint(1000, 5000)
        self.gradient_norm = random.uniform(0.5, 1.0)
        self.reputation = 1.0
        self.update_count = 0

        # Specialization affects base accuracy
        if config.data_distribution == "non-iid":
            # This bank specializes in one phishing type
            self.base_accuracy = 0.7 + random.random() * 0.15
        else:
            self.base_accuracy = 0.8 + random.random() * 0.1

    def train_round(self, round_num: int, global_accuracy: float, is_malicious: bool = False) -> Dict:
        """Simulate one training round for this bank."""
        if self.config.is_malicious and is_malicious:
            # Malicious bank sends bad updates
            self.status = "malicious"
            # Pretend to train normally but actually poison
            reported_accuracy = self.base_accuracy + min(0.15, round_num * 0.01)
            actual_accuracy = 0.3  # Poisoned
            self.gradient_norm = 2.0  # Large norm
        else:
            # Normal training
            self.status = "training"
            # Converge towards base accuracy
            target = self.base_accuracy
            progress = min(1.0, round_num / 15.0)
            noise = random.gauss(0, 0.02)
            self.accuracy = target * progress + 0.5 * (1 - progress) + noise
            self.accuracy = min(0.98, max(0.6, self.accuracy))
            reported_accuracy = self.accuracy
            actual_accuracy = self.accuracy
            self.gradient_norm = random.uniform(0.5, 1.0)

        self.loss = 1.0 - self.accuracy
        self.update_count += 1

        return {
            "bank_id": self.config.bank_id,
            "round": round_num,
            "accuracy": float(self.accuracy),
            "loss": float(self.loss),
            "samples": self.samples,
            "status": self.status,
            "gradient_norm": float(self.gradient_norm),
            "reputation": float(self.reputation),
        }


class FederatedSimulator:
    """
    Simulates federated learning training for demo purposes.

    Generates realistic training progress with configurable scenarios.
    """

    def __init__(self, scenario: ScenarioConfig):
        """Initialize simulator with scenario configuration."""
        self.scenario = scenario
        self.current_round = 0
        self.is_running = False
        self.is_paused = False
        self.banks: Dict[int, BankState] = {}

        # Initialize banks
        for bank_config in scenario.banks:
            self.banks[bank_config.bank_id] = BankState(bank_config)

        # Global model state
        self.global_accuracy = 0.5
        self.global_loss = 0.5
        self.privacy_epsilon_spent = 0.0

        # Attack state
        self.attack_detected = False
        self.attack_round = scenario.attack_config.get("start_round", 999) if scenario.attack_config else 999
        self.defense_activated = False

        logger.info(f"Initialized simulator: {scenario.name}")

    async def start_training(self):
        """Start the training simulation."""
        self.is_running = True
        self.is_paused = False
        logger.info("Training started")

    async def pause_training(self):
        """Pause the training simulation."""
        self.is_paused = True
        logger.info("Training paused")

    async def resume_training(self):
        """Resume the training simulation."""
        self.is_paused = False
        logger.info("Training resumed")

    async def reset_training(self):
        """Reset to initial state."""
        self.current_round = 0
        self.is_running = False
        self.is_paused = False
        self.attack_detected = False
        self.defense_activated = False

        # Reset banks
        for bank_config in self.scenario.banks:
            self.banks[bank_config.bank_id] = BankState(bank_config)

        self.global_accuracy = 0.5
        self.global_loss = 0.5
        self.privacy_epsilon_spent = 0.0

        logger.info("Training reset")

    def add_bank(self, bank_config: BankConfig):
        """Add a new bank to the federation."""
        self.banks[bank_config.bank_id] = BankState(bank_config)
        logger.info(f"Added bank {bank_config.bank_id}: {bank_config.name}")

    def remove_bank(self, bank_id: int):
        """Remove a bank from the federation."""
        if bank_id in self.banks:
            del self.banks[bank_id]
            logger.info(f"Removed bank {bank_id}")

    def set_bank_malicious(self, bank_id: int, is_malicious: bool):
        """Mark a bank as malicious or benign."""
        if bank_id in self.banks:
            self.banks[bank_id].config.is_malicious = is_malicious
            logger.info(f"Set bank {bank_id} malicious={is_malicious}")

    async def run_round(self) -> Dict:
        """Run one training round and return results."""
        if not self.is_running or self.is_paused:
            return None

        self.current_round += 1

        # Check for attack activation
        is_attack_round = self.current_round == self.attack_round

        # Train each bank
        bank_updates = []
        total_accuracy = 0.0
        total_samples = 0

        for bank_id, bank in self.banks.items():
            update = bank.train_round(
                self.current_round,
                self.global_accuracy,
                is_malicious=is_attack_round and bank.config.is_malicious
            )
            bank_updates.append(update)
            total_accuracy += update["accuracy"] * update["samples"]
            total_samples += update["samples"]

        # Compute global accuracy (weighted by samples)
        if total_samples > 0:
            self.global_accuracy = total_accuracy / total_samples
            self.global_loss = 1.0 - self.global_accuracy

        # Check for attack and activate defense
        if is_attack_round:
            self.attack_detected = True
            self.defense_activated = True

            # Update reputations (malicious bank gets downweighted)
            for bank in self.banks.values():
                if bank.config.is_malicious:
                    bank.reputation = max(0.1, bank.reputation - 0.3)
                else:
                    bank.reputation = min(1.0, bank.reputation + 0.05)

        # Update privacy budget
        self.privacy_epsilon_spent += self.scenario.epsilon

        # Check if training is complete
        is_complete = self.current_round >= self.scenario.num_rounds
        if is_complete:
            self.is_running = False
            for bank in self.banks.values():
                bank.status = "complete"

        # Prepare result
        result = {
            "type": "round_update",
            "round": self.current_round,
            "is_complete": is_complete,
            "global_accuracy": float(self.global_accuracy),
            "global_loss": float(self.global_loss),
            "banks": bank_updates,
            "privacy": {
                "epsilon_spent": float(self.privacy_epsilon_spent),
                "epsilon_limit": float(self.scenario.epsilon * self.scenario.num_rounds),
                "privacy_level": self.scenario.privacy_level,
                "encryption_active": self.scenario.privacy_level >= 2,
                "tee_mode": self.scenario.privacy_level >= 3,
            },
            "security": {
                "attack_detected": self.attack_detected,
                "defense_activated": self.defense_activated,
                "malicious_bank_id": next(
                    (b.bank_id for b in self.scenario.banks if b.is_malicious),
                    None
                ),
            },
            "scenario": {
                "name": self.scenario.name,
                "total_rounds": self.scenario.num_rounds,
            },
        }

        logger.debug(f"Round {self.current_round} complete: accuracy={self.global_accuracy:.4f}")

        return result

    def get_status(self) -> Dict:
        """Get current simulation status."""
        return {
            "scenario": self.scenario.name,
            "round": self.current_round,
            "total_rounds": self.scenario.num_rounds,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "global_accuracy": float(self.global_accuracy),
            "privacy_epsilon_spent": float(self.privacy_epsilon_spent),
            "num_banks": len(self.banks),
            "banks": [
                {
                    "bank_id": bank.config.bank_id,
                    "name": bank.config.name,
                    "location": bank.config.location,
                    "accuracy": float(bank.accuracy),
                    "status": bank.status,
                    "reputation": float(bank.reputation),
                    "is_malicious": bank.config.is_malicious,
                }
                for bank in self.banks.values()
            ],
        }

    def inject_attack(self, bank_id: int, attack_type: str = "sign_flip"):
        """Inject an attack from a specific bank."""
        if bank_id in self.banks:
            self.banks[bank_id].config.is_malicious = True
            self.attack_round = self.current_round + 1
            logger.info(f"Attack injected: bank {bank_id}, type {attack_type}")

    def update_privacy_level(self, level: int):
        """Update privacy level."""
        self.scenario.privacy_level = level
        logger.info(f"Privacy level updated to {level}")
