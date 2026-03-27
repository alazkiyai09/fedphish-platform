"""
ZK proof verification on server side.

Verifies gradient bound proofs from clients.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedphish.security.zkp import BatchProof, GradientBoundsProof, ZKVerifier

logger = logging.getLogger(__name__)


class ProofVerifier:
    """
    Verify ZK proofs from clients.

    Server-side component for proof verification.
    """

    def __init__(
        self,
        max_gradient_norm: float = 1.0,
        require_proofs: bool = True,
    ):
        """
        Initialize proof verifier.

        Args:
            max_gradient_norm: Expected maximum gradient norm
            require_proofs: Whether proofs are required
        """
        self.max_gradient_norm = max_gradient_norm
        self.require_proofs = require_proofs

        self.verifier = ZKVerifier(max_gradient_norm)
        self.verification_history = []

        logger.info(
            f"Initialized proof verifier: max_norm={max_gradient_norm}, "
            f"require_proofs={require_proofs}"
        )

    def verify_batch(
        self,
        proofs: List[Optional[GradientBoundsProof]],
        client_ids: List[int],
    ) -> Tuple[List[bool], List[int]]:
        """
        Verify batch of proofs.

        Args:
            proofs: List of proofs (None if client didn't send proof)
            client_ids: Client IDs

        Returns:
            Tuple of (is_valid_list, invalid_client_ids)
        """
        if not self.require_proofs:
            # All valid if proofs not required
            return [True] * len(proofs), []

        is_valid_list = []
        invalid_clients = []

        for proof, client_id in zip(proofs, client_ids):
            if proof is None:
                # No proof provided
                is_valid_list.append(False)
                invalid_clients.append(client_id)
                continue

            # Verify proof
            is_valid = self.verifier.verify_proof(proof)
            is_valid_list.append(is_valid)

            if not is_valid:
                invalid_clients.append(client_id)

            # Record verification
            self.verification_history.append({
                "client_id": client_id,
                "valid": is_valid,
                "proof_type": type(proof).__name__,
            })

            logger.debug(
                f"Verified proof from client {client_id}: valid={is_valid}"
            )

        return is_valid_list, invalid_clients

    def verify_aggregated_proof(
        self,
        batch_proof: BatchProof,
    ) -> bool:
        """
        Verify aggregated batch proof.

        Args:
            batch_proof: Batch proof

        Returns:
            True if valid
        """
        is_valid = self.verifier.verify_batch_proof(batch_proof)

        self.verification_history.append({
            "valid": is_valid,
            "proof_type": "batch",
            "num_proofs": batch_proof.proof_data.get("num_proofs", 0),
        })

        logger.info(f"Verified batch proof: valid={is_valid}")

        return is_valid

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        if not self.verification_history:
            return {}

        total = len(self.verification_history)
        valid = sum(1 for h in self.verification_history if h["valid"])

        return {
            "total_verifications": total,
            "valid_proofs": valid,
            "invalid_proofs": total - valid,
            "validity_rate": valid / total if total > 0 else 0,
        }


class ProofValidator:
    """
    Validate proofs and filter invalid updates.

    Combines proof verification with update filtering.
    """

    def __init__(
        self,
        max_gradient_norm: float = 1.0,
        require_proofs: bool = True,
    ):
        """
        Initialize proof validator.

        Args:
            max_gradient_norm: Maximum gradient norm
            require_proofs: Whether proofs are required
        """
        self.verifier = ProofVerifier(max_gradient_norm, require_proofs)

    def validate_and_filter(
        self,
        updates: List[np.ndarray],
        proofs: List[Optional[GradientBoundsProof]],
        client_ids: List[int],
    ) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Validate proofs and filter updates.

        Args:
            updates: Client updates
            proofs: Proofs for each update
            client_ids: Client IDs

        Returns:
            Tuple of (filtered_updates, valid_client_ids, invalid_client_ids)
        """
        # Verify proofs
        is_valid_list, invalid_clients = self.verifier.verify_batch(
            proofs, client_ids
        )

        # Filter updates
        valid_updates = []
        valid_clients = []

        for update, is_valid, client_id in zip(updates, is_valid_list, client_ids):
            if is_valid:
                valid_updates.append(update)
                valid_clients.append(client_id)

        logger.info(
            f"Filtered updates: {len(valid_updates)}/{len(updates)} valid, "
            f"{len(invalid_clients)} invalid"
        )

        return valid_updates, valid_clients, invalid_clients

    def get_validator_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return self.verifier.get_verification_stats()
