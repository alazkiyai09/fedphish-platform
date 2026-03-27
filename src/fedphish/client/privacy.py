"""
Client-side privacy engine.

Applies DP and HE before sending updates to server.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedphish.privacy.dp import DifferentialPrivacy, GradientClipper
from fedphish.privacy.he import EncryptedGradient, TenSEALContext, he_available
from fedphish.privacy.ht2ml import PrivacyLevel

logger = logging.getLogger(__name__)


class ClientPrivacyEngine:
    """
    Client-side privacy engine.

    Applies DP noise, gradient clipping, and HE encryption.
    """

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.LEVEL_3,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clipping_norm: float = 1.0,
        he_context: Optional[TenSEALContext] = None,
    ):
        """
        Initialize client privacy engine.

        Args:
            privacy_level: Privacy level to use
            epsilon: DP epsilon
            delta: DP delta
            clipping_norm: Gradient clipping norm
            he_context: TenSEAL context (created if None)
        """
        self.privacy_level = privacy_level
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_norm = clipping_norm

        # Initialize DP
        self.dp = DifferentialPrivacy(
            epsilon=epsilon,
            delta=delta,
            clipping_norm=clipping_norm,
        )
        self.clipper = GradientClipper(clipping_norm=clipping_norm)

        # Initialize HE if using level 2 or 3
        self.he_context = he_context
        self.use_he = privacy_level in [PrivacyLevel.LEVEL_2, PrivacyLevel.LEVEL_3]

        if self.use_he:
            if not he_available():
                logger.warning("HE not available, falling back to level 1")
                self.use_he = False
                self.privacy_level = PrivacyLevel.LEVEL_1
            elif self.he_context is None:
                self.he_context = TenSEALContext()

        logger.info(
            f"Initialized client privacy engine: level={privacy_level.name}, "
            f"use_he={self.use_he}"
        )

    def compute_private_update(
        self,
        gradients: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], Optional[List[EncryptedGradient]]]:
        """
        Compute privacy-preserving update.

        Args:
            gradients: Raw gradients from training

        Returns:
            Tuple of (plain_gradients, encrypted_gradients)
        """
        # Apply gradient clipping and DP noise
        private_grads = []
        for grad in gradients:
            clipped, clip_factor, norm = self.clipper.clip_gradients(grad)
            noisy = self.dp.add_noise(clipped)
            private_grads.append(noisy)

        # Encrypt if using HE
        encrypted_grads = None
        if self.use_he:
            encrypted_grads = [
                EncryptedGradient(self.he_context.context, grad)
                for grad in private_grads
            ]

        logger.debug(
            f"Computed private update: level={self.privacy_level.name}, "
            f"num_grads={len(private_grads)}, encrypted={encrypted_grads is not None}"
        )

        return private_grads, encrypted_grads

    def get_privacy_cost(self) -> Dict[str, float]:
        """Get current privacy cost."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "privacy_level": self.privacy_level.value,
        }


class PrivacyLevelConfig:
    """Configuration for different privacy levels."""

    LEVEL_1_CONFIG = {
        "privacy_level": PrivacyLevel.LEVEL_1,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clipping_norm": 1.0,
        "use_he": False,
    }

    LEVEL_2_CONFIG = {
        "privacy_level": PrivacyLevel.LEVEL_2,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clipping_norm": 1.0,
        "use_he": True,
    }

    LEVEL_3_CONFIG = {
        "privacy_level": PrivacyLevel.LEVEL_3,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clipping_norm": 1.0,
        "use_he": True,
        "use_tee": True,
    }

    @classmethod
    def get_config(cls, level: int) -> Dict[str, Any]:
        """Get configuration for privacy level."""
        if level == 1:
            return cls.LEVEL_1_CONFIG.copy()
        elif level == 2:
            return cls.LEVEL_2_CONFIG.copy()
        elif level == 3:
            return cls.LEVEL_3_CONFIG.copy()
        else:
            raise ValueError(f"Unknown privacy level: {level}")
