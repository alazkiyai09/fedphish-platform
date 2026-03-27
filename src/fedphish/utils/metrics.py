"""
Metrics for federated phishing detection.

Computes FL-specific, privacy, security, and detection metrics.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class FederatedMetrics:
    """Metrics specific to federated learning."""

    def __init__(self):
        """Initialize federated metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.round_metrics = []

    def log_round(
        self,
        round_num: int,
        num_clients: int,
        accuracy: float,
        loss: float,
        communication_cost: float,
        convergence_rate: float,
        training_time: float,
    ) -> Dict[str, Any]:
        """
        Log metrics for a training round.

        Args:
            round_num: Current round number
            num_clients: Number of clients participated
            accuracy: Validation accuracy
            loss: Validation loss
            communication_cost: Total bytes communicated
            convergence_rate: Rate of parameter change
            training_time: Time taken for round

        Returns:
            Dictionary of round metrics
        """
        round_data = {
            "round": round_num,
            "num_clients": num_clients,
            "accuracy": accuracy,
            "loss": loss,
            "communication_cost_mb": communication_cost / 1e6,
            "convergence_rate": convergence_rate,
            "training_time_sec": training_time,
        }

        self.round_metrics.append(round_data)
        return round_data

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all rounds."""
        if not self.round_metrics:
            return {}

        metrics = {
            "total_rounds": len(self.round_metrics),
            "final_accuracy": self.round_metrics[-1]["accuracy"],
            "final_loss": self.round_metrics[-1]["loss"],
            "best_accuracy": max(m["accuracy"] for m in self.round_metrics),
            "total_communication_mb": sum(
                m["communication_cost_mb"] for m in self.round_metrics
            ),
            "total_training_time_sec": sum(
                m["training_time_sec"] for m in self.round_metrics
            ),
            "avg_clients_per_round": np.mean(
                [m["num_clients"] for m in self.round_metrics]
            ),
        }

        return metrics


class PrivacyMetrics:
    """Track privacy-related metrics."""

    def __init__(self):
        """Initialize privacy metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.privacy_ledger = []
        self.epsilon_history = []
        self.delta_history = []

    def compute_epsilon_delta(
        self,
        noise_multiplier: float,
        sampling_probability: float,
        num_steps: int,
        delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute epsilon and delta for DP.

        Uses moments accountant approach (simplified).

        Args:
            noise_multiplier: Standard deviation of noise relative to sensitivity
            sampling_probability: Probability of sampling each record
            num_steps: Number of training steps
            delta: Target delta (computed if not provided)

        Returns:
            Tuple of (epsilon, delta)
        """
        # Simplified computation - in practice use advanced composition
        if delta is None:
            delta = 1e-5

        # Approximate epsilon using moments accountant
        epsilon = np.sqrt(2 * num_steps * np.log(1.25 / delta)) / noise_multiplier

        return epsilon, delta

    def log_privacy_spend(
        self,
        round_num: int,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        clipping_norm: float,
    ) -> Dict[str, Any]:
        """
        Log privacy spend for a round.

        Args:
            round_num: Round number
            epsilon: Privacy parameter
            delta: Delta parameter
            noise_multiplier: Noise multiplier used
            clipping_norm: Gradient clipping norm

        Returns:
            Dictionary of privacy metrics
        """
        privacy_data = {
            "round": round_num,
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "clipping_norm": clipping_norm,
        }

        self.privacy_ledger.append(privacy_data)
        self.epsilon_history.append(epsilon)
        self.delta_history.append(delta)

        return privacy_data

    def get_total_privacy_cost(self) -> Dict[str, float]:
        """Get total privacy cost across all rounds."""
        if not self.privacy_ledger:
            return {"epsilon": 0.0, "delta": 0.0}

        # Compose privacy parameters (advanced composition)
        total_epsilon = sum(self.epsilon_history)
        total_delta = sum(self.delta_history)

        return {
            "epsilon": total_epsilon,
            "delta": total_delta,
        }

    def compute_privacy_loss(
        self,
        gradients: np.ndarray,
        noise_multiplier: float,
        clipping_norm: float,
    ) -> float:
        """
        Compute actual privacy loss from gradients.

        Args:
            gradients: Gradient values
            noise_multiplier: Noise multiplier
            clipping_norm: Clipping norm

        Returns:
            Estimated privacy loss
        """
        # Compute gradient norms
        gradient_norms = np.linalg.norm(gradients, axis=1)

        # Fraction of clipped gradients
        clipped_fraction = np.mean(gradient_norms > clipping_norm)

        return clipped_fraction


class SecurityMetrics:
    """Track security-related metrics."""

    def __init__(self):
        """Initialize security metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.proof_verification_times = []
        self.proof_validity_rates = []
        self.attack_detections = []
        self.reputation_scores = {}

    def log_proof_verification(
        self,
        round_num: int,
        verification_time: float,
        num_proofs: int,
        num_valid: int,
    ) -> Dict[str, Any]:
        """
        Log ZK proof verification metrics.

        Args:
            round_num: Round number
            verification_time: Time taken to verify proofs
            num_proofs: Total number of proofs
            num_valid: Number of valid proofs

        Returns:
            Dictionary of verification metrics
        """
        validity_rate = num_valid / num_proofs if num_proofs > 0 else 0

        metrics = {
            "round": round_num,
            "verification_time_sec": verification_time,
            "num_proofs": num_proofs,
            "num_valid": num_valid,
            "validity_rate": validity_rate,
            "avg_verification_time_ms": (
                verification_time / num_proofs * 1000 if num_proofs > 0 else 0
            ),
        }

        self.proof_verification_times.append(verification_time)
        self.proof_validity_rates.append(validity_rate)

        return metrics

    def log_attack_detection(
        self,
        round_num: int,
        num_malicious: int,
        num_detected: int,
        num_false_positives: int,
        defense_strategy: str,
    ) -> Dict[str, Any]:
        """
        Log attack detection metrics.

        Args:
            round_num: Round number
            num_malicious: Actual number of malicious clients
            num_detected: Number detected as malicious
            num_false_positives: Number of false positives
            defense_strategy: Defense strategy used

        Returns:
            Dictionary of detection metrics
        """
        true_positives = min(num_malicious, num_detected)
        false_negatives = max(0, num_malicious - num_detected)

        precision = (
            true_positives / (true_positives + num_false_positives)
            if (true_positives + num_false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = {
            "round": round_num,
            "num_malicious": num_malicious,
            "num_detected": num_detected,
            "num_false_positives": num_false_positives,
            "defense_strategy": defense_strategy,
            "true_positive_rate": recall,
            "false_positive_rate": (
                num_false_positives / num_detected if num_detected > 0 else 0
            ),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        self.attack_detections.append(metrics)

        return metrics

    def update_reputation(
        self,
        client_id: int,
        reputation_score: float,
    ):
        """
        Update reputation score for a client.

        Args:
            client_id: Client ID
            reputation_score: New reputation score
        """
        self.reputation_scores[client_id] = reputation_score

    def get_reputation_summary(self) -> Dict[str, Any]:
        """Get summary of reputation scores."""
        if not self.reputation_scores:
            return {}

        scores = list(self.reputation_scores.values())

        return {
            "num_clients": len(self.reputation_scores),
            "mean_reputation": np.mean(scores),
            "std_reputation": np.std(scores),
            "min_reputation": np.min(scores),
            "max_reputation": np.max(scores),
        }


class DetectionMetrics:
    """Standard detection/classification metrics."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive detection metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # Add AUC if probabilities provided
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)

        return metrics

    @staticmethod
    def compute_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return fpr, tpr, thresholds

    @staticmethod
    def compute_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return precision, recall, thresholds


class MetricsTracker:
    """Combined metrics tracker for all metric types."""

    def __init__(self):
        """Initialize comprehensive metrics tracker."""
        self.federated = FederatedMetrics()
        self.privacy = PrivacyMetrics()
        self.security = SecurityMetrics()

    def reset(self):
        """Reset all metrics."""
        self.federated.reset()
        self.privacy.reset()
        self.security.reset()

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        return {
            "federated": self.federated.get_summary(),
            "privacy": self.privacy.get_total_privacy_cost(),
            "security": {
                "avg_proof_verification_time": (
                    np.mean(self.security.proof_verification_times)
                    if self.security.proof_verification_times
                    else 0
                ),
                "avg_proof_validity_rate": (
                    np.mean(self.security.proof_validity_rates)
                    if self.security.proof_validity_rates
                    else 0
                ),
                "reputation": self.security.get_reputation_summary(),
            },
        }
