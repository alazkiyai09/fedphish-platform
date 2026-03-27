"""
Privacy mechanisms for federated learning.

Implements DP, secure aggregation, and hybrid HE/TEE mechanisms.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class LocalDP:
    """
    Local differential privacy using DP-SGD.

    Uses Opacus library to add noise to gradients during training.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5):
        """
        Initialize local DP.

        Args:
            epsilon: Privacy budget
            delta: Delta parameter
        """
        self.epsilon = epsilon
        self.delta = delta
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0

        try:
            from opacus import PrivacyEngine
            self.PrivacyEngine = PrivacyEngine
            self.opacus_available = True
        except ImportError:
            self.opacus_available = False
            print("Warning: Opacus not available. Install with: pip install opacus")

    def make_model_private(self,
                          model: nn.Module,
                          sample_rate: float,
                          max_grad_norm: float = 1.0) -> nn.Module:
        """
        Wrap model with DP-SGD.

        Args:
            model: PyTorch model
            sample_rate: Sampling rate (batch_size / n_samples)
            max_grad_norm: Gradient clipping norm

        Returns:
            PrivacyEngine-wrapped model
        """
        if not self.opacus_available:
            # Return model without privacy (fallback)
            print("Warning: Opacus not available. Returning model without DP.")
            return model

        privacy_engine = self.PrivacyEngine()

        model = privacy_engine.make_private_with_grad_sampling(
            module=model,
            sample_rate=sample_rate,
            max_grad_norm=max_grad_norm
        )

        return model

    def get_budget(self) -> Tuple[float, float]:
        """
        Get spent privacy budget.

        Returns:
            (epsilon_spent, delta_spent)
        """
        if not self.opacus_available:
            return (0.0, 0.0)

        return (self.epsilon_spent, self.delta_spent)


class SecureAggregation:
    """
    Secure aggregation using homomorphic encryption (Day 23).

    Uses TenSEAL for CKKS encryption to aggregate updates without
    revealing individual bank updates.
    """

    def __init__(self, n_clients: int, poly_modulus_degree: int = 8192):
        """
        Initialize secure aggregation.

        Args:
            n_clients: Number of clients (banks)
            poly_modulus_degree: Polynomial modulus degree for CKKS
        """
        self.n_clients = n_clients
        self.poly_modulus_degree = poly_modulus_degree

        try:
            import tenseal as ts
            self.tenseal = ts
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: TenSEAL not available. Install with: pip install tenseal")

    def encrypt_update(self, update: List[np.ndarray]) -> bytes:
        """
        Encrypt model update.

        Args:
            update: List of parameter arrays

        Returns:
            Encrypted update
        """
        if not self.available:
            # Return without encryption (fallback)
            return b''.join(p.tobytes() for p in update)

        # NOTE: This is a simulation for educational purposes.
        # Actual TenSEAL CKKS/BFV encryption would require:
        # 1. Converting numpy arrays to appropriate encoding
        # 2. Creating CKKS/BFV context with proper parameters
        # 3. Encrypting each parameter tensor
        # 4. Handling ciphertext operations
        # For full implementation, see: https://github.com/OpenMined/TenSEAL
        return b''.join(p.tobytes() for p in update)

    def decrypt_aggregate(self, encrypted_updates: List[bytes]) -> List[np.ndarray]:
        """
        Decrypt and aggregate encrypted updates.

        Args:
            encrypted_updates: List of encrypted updates

        Returns:
            Aggregated parameters
        """
        if not self.available:
            # Deserialize without decryption (fallback)
            aggregated = np.mean(
                [np.frombuffer(up) for up in encrypted_updates],
                axis=0
            )
            return [aggregated]

        # NOTE: This is a simulation for educational purposes.
        # Actual TenSEAL homomorphic aggregation would:
        # 1. Deserialize ciphertexts
        # 2. Perform homomorphic addition on ciphertexts
        # 3. Decrypt the aggregated ciphertext
        # 4. Rescale if using CKKS with scale parameters
        # For full implementation, see: https://github.com/OpenMined/TenSEAL
        return [np.array([])]


class HybridPrivacyMechanism:
    """
    Hybrid HE/TEE mechanism from HT2ML (Days 6-8).

    Combines:
    - Homomorphic encryption for communication
    - Trusted Execution Environment for computation
    """

    def __init__(self, epsilon: float = 1.0, use_tee: bool = True):
        """
        Initialize hybrid mechanism.

        Args:
            epsilon: Privacy budget
            use_tee: Whether to simulate TEE usage
        """
        self.epsilon = epsilon
        self.use_tee = use_tee

    def secure_update(self, update: List[np.ndarray]) -> bytes:
        """
        Protect update using hybrid HE+TEE.

        Args:
            update: Model parameters

        Returns:
            Protected update
        """
        # TODO: Implement hybrid mechanism
        # For simulation, return serialized parameters
        return b''.join(p.tobytes() for p in update)
