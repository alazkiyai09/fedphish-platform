"""
Client-side ZK proof generation.

Generates zero-knowledge proofs for gradient bounds.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedphish.security.zkp import GradientBoundsProof, ZKProver

logger = logging.getLogger(__name__)


class GradientBoundsProver:
    """
    Generate ZK proofs for gradient bounds.

    Client-side component for proof generation.
    """

    def __init__(
        self,
        max_gradient_norm: float = 1.0,
    ):
        """
        Initialize gradient bounds prover.

        Args:
            max_gradient_norm: Maximum allowed gradient norm
        """
        self.max_gradient_norm = max_gradient_norm
        self.prover = ZKProver(max_gradient_norm)

        logger.info(f"Initialized gradient bounds prover: max_norm={max_gradient_norm}")

    def generate_proof(
        self,
        gradients: List[np.ndarray],
    ) -> GradientBoundsProof:
        """
        Generate proof for gradient bounds.

        Args:
            gradients: Gradient values

        Returns:
            Gradient bounds proof
        """
        # Combine gradients
        combined = np.concatenate([g.flatten() for g in gradients])

        # Generate proof
        proof = self.prover.generate_proof(combined)

        logger.debug("Generated ZK proof for gradient bounds")

        return proof

    def generate_batch_proof(
        self,
        gradients_list: List[List[np.ndarray]],
    ) -> Any:
        """
        Generate batch proof for multiple rounds.

        Args:
            gradients_list: List of gradient batches

        Returns:
            Batch proof
        """
        # Flatten each batch
        flattened = [
            np.concatenate([g.flatten() for g in grads])
            for grads in gradients_list
        ]

        # Generate batch proof
        batch_proof = self.prover.generate_batch_proof(flattened)

        logger.debug(f"Generated batch ZK proof for {len(gradients_list)} rounds")

        return batch_proof


class ProofGenerator:
    """
    High-level proof generator for clients.

    Orchestrates proof generation for FL updates.
    """

    def __init__(
        self,
        max_gradient_norm: float = 1.0,
        enable_proofs: bool = True,
    ):
        """
        Initialize proof generator.

        Args:
            max_gradient_norm: Maximum gradient norm
            enable_proofs: Whether to generate proofs
        """
        self.max_gradient_norm = max_gradient_norm
        self.enable_proofs = enable_proofs

        if enable_proofs:
            self.prover = GradientBoundsProver(max_gradient_norm)

        logger.info(f"Initialized proof generator: enabled={enable_proofs}")

    def generate_update_with_proof(
        self,
        gradients: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], Optional[GradientBoundsProof]]:
        """
        Generate update with ZK proof.

        Args:
            gradients: Gradient updates

        Returns:
            Tuple of (gradients, proof)
        """
        if not self.enable_proofs:
            return gradients, None

        # Generate proof
        proof = self.prover.generate_proof(gradients)

        return gradients, proof

    def verify_gradients_locally(
        self,
        gradients: List[np.ndarray],
    ) -> Tuple[bool, float]:
        """
        Verify gradients are within bounds locally.

        Args:
            gradients: Gradient values

        Returns:
            Tuple of (is_within_bounds, actual_norm)
        """
        # Combine and compute norm
        combined = np.concatenate([g.flatten() for g in gradients])
        norm = np.linalg.norm(combined)

        is_valid = norm <= self.max_gradient_norm

        return is_valid, norm
