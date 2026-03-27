"""
Attack implementations for federated learning.

Implements various Byzantine and poisoning attacks for testing defenses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ByzantineAttack:
    """Base class for Byzantine attacks."""

    def __init__(
        self,
        num_malicious: int = 1,
        attack_strength: float = 1.0,
    ):
        """
        Initialize attack.

        Args:
            num_malicious: Number of malicious clients
            attack_strength: Strength of attack (multiplicative factor)
        """
        self.num_malicious = num_malicious
        self.attack_strength = attack_strength

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """
        Execute attack on gradients.

        Args:
            gradients: List of gradients from all clients
            malicious_indices: Indices of malicious clients

        Returns:
            Modified gradients with attack
        """
        raise NotImplementedError


class SignFlipAttack(ByzantineAttack):
    """
    Sign flipping attack.

    Malicious clients send negated gradients.
    """

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """
        Execute sign flip attack.

        Args:
            gradients: List of gradients
            malicious_indices: Indices of malicious clients

        Returns:
            Gradients with sign flips
        """
        attacked_gradients = gradients.copy()

        for idx in malicious_indices:
            if idx < len(attacked_gradients):
                # Flip signs
                attacked_gradients[idx] = -self.attack_strength * attacked_gradients[idx]
                logger.debug(f"Applied sign flip to client {idx}")

        return attacked_gradients


class GaussianNoiseAttack(ByzantineAttack):
    """
    Gaussian noise attack.

    Malicious clients send gradients with large random noise.
    """

    def __init__(
        self,
        num_malicious: int = 1,
        attack_strength: float = 1.0,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ):
        """
        Initialize Gaussian noise attack.

        Args:
            num_malicious: Number of malicious clients
            attack_strength: Attack strength
            noise_mean: Mean of noise
            noise_std: Standard deviation of noise
        """
        super().__init__(num_malicious, attack_strength)
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Execute Gaussian noise attack."""
        attacked_gradients = gradients.copy()

        for idx in malicious_indices:
            if idx < len(attacked_gradients):
                # Add Gaussian noise
                noise = np.random.normal(
                    self.noise_mean,
                    self.noise_std * self.attack_strength,
                    attacked_gradients[idx].shape,
                )
                attacked_gradients[idx] = attacked_gradients[idx] + noise
                logger.debug(f"Applied Gaussian noise to client {idx}")

        return attacked_gradients


class BackdoorAttack(ByzantineAttack):
    """
    Backdoor attack.

    Malicious clients attempt to embed backdoor in model.
    """

    def __init__(
        self,
        num_malicious: int = 1,
        attack_strength: float = 1.0,
        trigger_pattern: Optional[np.ndarray] = None,
        target_label: int = 1,
    ):
        """
        Initialize backdoor attack.

        Args:
            num_malicious: Number of malicious clients
            attack_strength: Attack strength
            trigger_pattern: Pattern to trigger backdoor
            target_label: Target label for backdoor
        """
        super().__init__(num_malicious, attack_strength)
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Execute backdoor attack."""
        attacked_gradients = gradients.copy()

        for idx in malicious_indices:
            if idx < len(attacked_gradients):
                # Add backdoor gradient (push towards target)
                backdoor_grad = self._create_backdoor_gradient(attacked_gradients[idx].shape)
                attacked_gradients[idx] = (
                    attacked_gradients[idx] +
                    self.attack_strength * backdoor_grad
                )
                logger.debug(f"Applied backdoor to client {idx}")

        return attacked_gradients

    def _create_backdoor_gradient(self, shape: np.ndarray) -> np.ndarray:
        """Create gradient for backdoor."""
        # Simplified: create gradient that pushes towards target
        backdoor = np.random.randn(*shape) * 0.1

        # If trigger pattern provided, shape it accordingly
        if self.trigger_pattern is not None:
            # Scale by trigger pattern
            backdoor = backdoor * self.trigger_pattern

        return backdoor


class LabelFlippingAttack(ByzantineAttack):
    """
    Label flipping attack (data poisoning).

    Malicious clients flip labels during training.
    Note: This requires modifying the training process itself.
    """

    def __init__(
        self,
        num_malicious: int = 1,
        flip_ratio: float = 0.5,
        target_class: Optional[int] = None,
    ):
        """
        Initialize label flipping attack.

        Args:
            num_malicious: Number of malicious clients
            flip_ratio: Ratio of labels to flip
            target_class: Target class to flip to (None = random)
        """
        super().__init__(num_malicious, attack_strength=1.0)
        self.flip_ratio = flip_ratio
        self.target_class = target_class

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """
        Simulate label flipping attack.

        Note: In practice, this would modify training data.
        Here we simulate by sending incorrect gradients.
        """
        attacked_gradients = gradients.copy()

        for idx in malicious_indices:
            if idx < len(attacked_gradients):
                # Send gradient that corresponds to flipped labels
                # Simulate by reversing gradient direction
                attacked_gradients[idx] = -attacked_gradients[idx] * self.flip_ratio
                logger.debug(
                    f"Simulated label flipping (ratio={self.flip_ratio}) "
                    f"for client {idx}"
                )

        return attacked_gradients


class AGRAttackResistant(ByzantineAttack):
    """
    Attack specifically designed against AGR (aggregation rule).

    Adapts to the aggregation strategy being used.
    """

    def __init__(
        self,
        num_malicious: int = 1,
        attack_strength: float = 1.0,
        aggregation_strategy: str = "fedavg",
    ):
        """
        Initialize AGR-resistant attack.

        Args:
            num_malicious: Number of malicious clients
            attack_strength: Attack strength
            aggregation_strategy: Target aggregation strategy
        """
        super().__init__(num_malicious, attack_strength)
        self.aggregation_strategy = aggregation_strategy

    def execute(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Execute AGR-resistant attack."""
        attacked_gradients = gradients.copy()

        if self.aggregation_strategy == "fedavg":
            # For FedAvg, make malicious updates cancel out benign ones
            attacked_gradients = self._attack_fedavg(
                attacked_gradients, malicious_indices
            )
        elif self.aggregation_strategy == "krum":
            # For Krum, make malicious updates appear close to each other
            attacked_gradients = self._attack_krum(
                attacked_gradients, malicious_indices
            )
        elif self.aggregation_strategy == "trimmed_mean":
            # For trimmed mean, push values to extremes
            attacked_gradients = self._attack_trimmed_mean(
                attacked_gradients, malicious_indices
            )
        else:
            # Default: sign flip
            attacked_gradients = SignFlipAttack(
                self.num_malicious, self.attack_strength
            ).execute(gradients, malicious_indices)

        return attacked_gradients

    def _attack_fedavg(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Attack optimized for FedAvg."""
        # Make malicious updates negatively proportional to benign count
        benign_count = len(gradients) - len(malicious_indices)
        scale_factor = -(benign_count / len(malicious_indices)) * self.attack_strength

        for idx in malicious_indices:
            if idx < len(gradients):
                # Compute average benign gradient
                benign_grads = [g for i, g in enumerate(gradients) if i not in malicious_indices]
                avg_benign = np.mean(benign_grads, axis=0)

                # Send malicious update to cancel out
                attacked = avg_benign * scale_factor
                gradients[idx] = attacked

        return gradients

    def _attack_krum(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Attack optimized for Krum."""
        # Krum selects update closest to all others
        # Make malicious updates close to each other but far from benign

        if len(malicious_indices) < 2:
            return gradients

        # Create malicious update that is far from benign
        benign_grads = [g for i, g in enumerate(gradients) if i not in malicious_indices]
        avg_benign = np.mean(benign_grads, axis=0)

        # Create malicious point far from benign
        malicious_point = avg_benign + 10 * self.attack_strength * np.random.randn(*avg_benign.shape)

        # All malicious clients send the same update
        for idx in malicious_indices:
            gradients[idx] = malicious_point.copy()

        return gradients

    def _attack_trimmed_mean(
        self,
        gradients: List[np.ndarray],
        malicious_indices: List[int],
    ) -> List[np.ndarray]:
        """Attack optimized for trimmed mean."""
        # Push values to extremes to affect trimmed mean

        benign_grads = [g for i, g in enumerate(gradients) if i not in malicious_indices]
        avg_benign = np.mean(benign_grads, axis=0)

        # Half of malicious clients push high, half push low
        for i, idx in enumerate(malicious_indices):
            if i % 2 == 0:
                gradients[idx] = avg_benign + 5 * self.attack_strength
            else:
                gradients[idx] = avg_benign - 5 * self.attack_strength

        return gradients


class AttackSimulator:
    """
    Simulate attacks in federated learning.

    Orchestrates attack execution and evaluation.
    """

    def __init__(
        self,
        attack_type: str = "sign_flip",
        num_malicious: int = 1,
        attack_strength: float = 1.0,
    ):
        """
        Initialize attack simulator.

        Args:
            attack_type: Type of attack
            num_malicious: Number of malicious clients
            attack_strength: Attack strength
        """
        self.attack_type = attack_type
        self.num_malicious = num_malicious
        self.attack_strength = attack_strength

        # Create attack instance
        self.attack = self._create_attack()

        logger.info(
            f"Initialized attack simulator: type={attack_type}, "
            f"num_malicious={num_malicious}"
        )

    def _create_attack(self) -> ByzantineAttack:
        """Create attack instance."""
        if self.attack_type == "sign_flip":
            return SignFlipAttack(self.num_malicious, self.attack_strength)
        elif self.attack_type == "gaussian_noise":
            return GaussianNoiseAttack(self.num_malicious, self.attack_strength)
        elif self.attack_type == "backdoor":
            return BackdoorAttack(self.num_malicious, self.attack_strength)
        elif self.attack_type == "label_flip":
            return LabelFlippingAttack(self.num_malicious)
        elif self.attack_type == "agr_resistant":
            return AGRAttackResistant(
                self.num_malicious,
                self.attack_strength,
                aggregation_strategy="fedavg",
            )
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def simulate_round(
        self,
        gradients: List[np.ndarray],
        malicious_indices: Optional[List[int]] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Simulate attack for one round.

        Args:
            gradients: Gradients from all clients
            malicious_indices: Indices of malicious clients (random if None)

        Returns:
            Tuple of (attacked_gradients, malicious_indices)
        """
        if malicious_indices is None:
            # Randomly select malicious clients
            num_clients = len(gradients)
            malicious_indices = np.random.choice(
                num_clients,
                size=min(self.num_malicious, num_clients),
                replace=False,
            ).tolist()

        # Execute attack
        attacked_gradients = self.attack.execute(gradients, malicious_indices)

        logger.info(
            f"Simulated {self.attack_type} attack: "
            f"{len(malicious_indices)} malicious clients"
        )

        return attacked_gradients, malicious_indices

    def evaluate_attack_success(
        self,
        original_accuracy: float,
        attacked_accuracy: float,
    ) -> Dict[str, float]:
        """
        Evaluate attack success.

        Args:
            original_accuracy: Accuracy without attack
            attacked_accuracy: Accuracy with attack

        Returns:
            Attack success metrics
        """
        accuracy_drop = original_accuracy - attacked_accuracy
        success_rate = accuracy_drop / original_accuracy if original_accuracy > 0 else 0

        return {
            "accuracy_drop": accuracy_drop,
            "success_rate": success_rate,
            "attack_successful": success_rate > 0.1,  # 10% drop threshold
        }


def create_attack(
    attack_type: str,
    num_malicious: int = 1,
    attack_strength: float = 1.0,
    **kwargs,
) -> ByzantineAttack:
    """
    Create attack instance.

    Args:
        attack_type: Type of attack
        num_malicious: Number of malicious clients
        attack_strength: Attack strength
        **kwargs: Additional attack-specific parameters

    Returns:
        Attack instance
    """
    simulator = AttackSimulator(attack_type, num_malicious, attack_strength)
    return simulator.attack
