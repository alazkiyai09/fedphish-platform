"""
Hybrid ensemble model for phishing detection.

Combines DistilBERT (for semantic understanding) and XGBoost (for engineered features).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from .transformer import DistilBERTWithLoRA, FeatureExtractor

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """
    Hybrid ensemble combining transformer and gradient boosting.

    Strategy:
    1. DistilBERT extracts semantic embeddings from text
    2. XGBoost uses engineered features
    3. Learnable combination weights
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        lora_rank: int = 8,
        xgb_params: Optional[Dict] = None,
        combination_method: str = "learned",  # 'learned', 'average', 'voting'
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize hybrid ensemble.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            lora_rank: LoRA rank for DistilBERT
            xgb_params: XGBoost hyperparameters
            combination_method: How to combine models
            device: Device for transformer
        """
        self.num_labels = num_labels
        self.combination_method = combination_method
        self.device = device

        # Initialize DistilBERT with LoRA
        self.transformer = DistilBERTWithLoRA(
            model_name=model_name,
            num_labels=num_labels,
            lora_rank=lora_rank,
            freeze_base=True,
        ).to(device)

        # Initialize XGBoost
        xgb_params = xgb_params or {
            "objective": "binary:logistic",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        self.xgb_model = xgb.XGBClassifier(**xgb_params)

        # Learnable combination weights
        if combination_method == "learned":
            self.combination_weights = nn.Parameter(
                torch.ones(2, device=device) / 2  # Start with equal weights
            )
        else:
            self.combination_weights = None

        # Feature extractor for embeddings
        self.feature_extractor = FeatureExtractor(model_name=model_name)

        self.is_fitted = False

        logger.info(
            f"Initialized hybrid ensemble: "
            f"model={model_name}, combination={combination_method}"
        )

    def fit(
        self,
        texts: List[str],
        engineered_features: np.ndarray,
        labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
    ):
        """
        Train both components of ensemble.

        Args:
            texts: Training texts
            engineered_features: Training engineered features
            labels: Training labels
            val_texts: Validation texts (optional)
            val_features: Validation engineered features (optional)
            val_labels: Validation labels (optional)
            num_epochs: Number of training epochs for transformer
            learning_rate: Learning rate for transformer
            batch_size: Batch size for transformer
        """
        logger.info("Training hybrid ensemble...")

        # Train XGBoost on engineered features
        logger.info("Training XGBoost component...")
        self.xgb_model.fit(
            engineered_features,
            labels,
            eval_set=(
                [val_features, val_labels]
                if val_features is not None and val_labels is not None
                else None
            ),
            verbose=False,
        )

        # Train DistilBERT with LoRA
        logger.info("Training DistilBERT component...")
        self._train_transformer(
            texts,
            labels,
            val_texts=val_texts,
            val_labels=val_labels,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        # Learn combination weights if using learned combination
        if self.combination_method == "learned" and val_texts is not None:
            logger.info("Learning combination weights...")
            self._learn_combination_weights(
                val_texts,
                val_features,
                val_labels,
            )

        self.is_fitted = True
        logger.info("Hybrid ensemble training complete")

    def _train_transformer(
        self,
        texts: List[str],
        labels: np.ndarray,
        val_texts: Optional[List[str]],
        val_labels: Optional[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
    ):
        """Train transformer component."""
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from torch.utils.data import DataLoader, TensorDataset

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(self.transformer.model_name)
        inputs = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Create dataset
        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Validation dataloader
        val_dataloader = None
        if val_texts is not None and val_labels is not None:
            val_inputs = tokenizer(
                val_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            val_dataset = TensorDataset(
                val_inputs["input_ids"],
                val_inputs["attention_mask"],
                torch.tensor(val_labels, dtype=torch.long),
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer (only LoRA parameters + classifier)
        optimizer = torch.optim.AdamW(
            list(self.transformer.get_lora_parameters()) +
            list(self.transformer.get_classifier_parameters()),
            lr=learning_rate,
        )

        # Training loop
        self.transformer.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask, labels_batch = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels_batch = labels_batch.to(self.device)

                # Forward
                logits = self.transformer(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels_batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Validation
            if val_dataloader is not None:
                val_acc = self._evaluate_transformer(val_dataloader)
                logger.info(f"Validation Accuracy: {val_acc:.4f}")

    def _evaluate_transformer(self, dataloader: DataLoader) -> float:
        """Evaluate transformer."""
        self.transformer.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels_batch = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels_batch = labels_batch.to(self.device)

                logits = self.transformer(input_ids, attention_mask)
                predictions = logits.argmax(dim=1)

                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

        self.transformer.train()
        return correct / total

    def _learn_combination_weights(
        self,
        val_texts: List[str],
        val_features: np.ndarray,
        val_labels: np.ndarray,
    ):
        """Learn optimal combination weights on validation set."""
        # Get predictions from both models
        transformer_probs = self._predict_transformer_proba(val_texts)
        xgb_probs = self.xgb_model.predict_proba(val_features)

        # Stack predictions
        stacked_probs = np.stack([transformer_probs, xgb_probs], axis=1)

        # Learn weights via grid search
        best_weight = 0.5
        best_accuracy = 0

        for weight in np.linspace(0, 1, 21):
            # Combined predictions
            combined_probs = (
                weight * transformer_probs +
                (1 - weight) * xgb_probs
            )
            combined_preds = combined_probs.argmax(axis=1)
            accuracy = (combined_preds == val_labels).mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight

        # Set learned weights
        self.combination_weights.data = torch.tensor(
            [best_weight, 1 - best_weight],
            device=self.device,
        )

        logger.info(
            f"Learned combination weights: "
            f"transformer={best_weight:.3f}, xgb={1-best_weight:.3f}"
        )

    def _predict_transformer_proba(self, texts: List[str]) -> np.ndarray:
        """Get probability predictions from transformer."""
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader, TensorDataset

        tokenizer = AutoTokenizer.from_pretrained(self.transformer.model_name)
        inputs = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=32)

        self.transformer.eval()
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits = self.transformer(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict(
        self,
        texts: List[str],
        engineered_features: np.ndarray,
    ) -> np.ndarray:
        """
        Make predictions with ensemble.

        Args:
            texts: Text inputs
            engineered_features: Engineered features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get predictions from both models
        transformer_probs = self._predict_transformer_proba(texts)
        xgb_probs = self.xgb_model.predict_proba(engineered_features)

        # Combine predictions
        if self.combination_method == "learned":
            # Use learned weights
            weights = torch.softmax(self.combination_weights, dim=0)
            combined_probs = (
                weights[0].item() * transformer_probs +
                weights[1].item() * xgb_probs
            )
        elif self.combination_method == "average":
            # Equal weights
            combined_probs = (transformer_probs + xgb_probs) / 2
        elif self.combination_method == "voting":
            # Majority voting
            transformer_preds = transformer_probs.argmax(axis=1)
            xgb_preds = xgb_probs.argmax(axis=1)
            combined_preds = np.round(
                (transformer_preds.astype(float) + xgb_preds.astype(float)) / 2
            ).astype(int)
            return combined_preds
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        return combined_probs.argmax(axis=1)

    def predict_proba(
        self,
        texts: List[str],
        engineered_features: np.ndarray,
    ) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            texts: Text inputs
            engineered_features: Engineered features

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        transformer_probs = self._predict_transformer_proba(texts)
        xgb_probs = self.xgb_model.predict_proba(engineered_features)

        if self.combination_method == "learned":
            weights = torch.softmax(self.combination_weights, dim=0)
            combined_probs = (
                weights[0].item() * transformer_probs +
                weights[1].item() * xgb_probs
            )
        else:
            combined_probs = (transformer_probs + xgb_probs) / 2

        return combined_probs

    def save(self, path: str):
        """Save ensemble model."""
        import pickle

        save_dict = {
            "transformer": self.transformer.state_dict(),
            "xgb_model": self.xgb_model,
            "combination_weights": (
                self.combination_weights.cpu().numpy()
                if self.combination_weights is not None
                else None
            ),
            "combination_method": self.combination_method,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"Saved ensemble model to {path}")

    def load(self, path: str):
        """Load ensemble model."""
        import pickle

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        self.transformer.load_state_dict(save_dict["transformer"])
        self.xgb_model = save_dict["xgb_model"]
        self.combination_method = save_dict["combination_method"]
        self.is_fitted = save_dict["is_fitted"]

        if save_dict["combination_weights"] is not None:
            self.combination_weights = nn.Parameter(
                torch.tensor(
                    save_dict["combination_weights"],
                    device=self.device,
                )
            )

        logger.info(f"Loaded ensemble model from {path}")


class WeightedCombiner:
    """
    Learn optimal combination weights for ensemble.

    Uses validation set to find weights that maximize accuracy.
    """

    def __init__(
        self,
        num_models: int = 2,
        weight_constraint: str = "simplex",  # 'simplex', 'none'
    ):
        """
        Initialize weight combiner.

        Args:
            num_models: Number of models to combine
            weight_constraint: Constraint on weights
        """
        self.num_models = num_models
        self.weight_constraint = weight_constraint

    def fit(
        self,
        predictions: List[np.ndarray],  # List of (n_samples, n_classes) arrays
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Learn optimal weights.

        Args:
            predictions: Predictions from each model
            labels: True labels

        Returns:
            Optimal weights
        """
        from scipy.optimize import minimize

        # Objective function (negative accuracy)
        def objective(weights):
            # Combine predictions
            combined = sum(w * pred for w, pred in zip(weights, predictions))
            combined_preds = combined.argmax(axis=1)
            accuracy = (combined_preds == labels).mean()
            return -accuracy

        # Constraints
        if self.weight_constraint == "simplex":
            # Weights sum to 1 and are non-negative
            constraints = {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1,
            }
            bounds = [(0, 1) for _ in range(self.num_models)]
            x0 = np.ones(self.num_models) / self.num_models
        else:
            constraints = None
            bounds = None
            x0 = np.ones(self.num_models) / self.num_models

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        self.weights = result.x
        logger.info(f"Learned ensemble weights: {self.weights}")

        return self.weights

    def combine(
        self,
        predictions: List[np.ndarray],
    ) -> np.ndarray:
        """
        Combine predictions using learned weights.

        Args:
            predictions: Predictions from each model

        Returns:
            Combined predictions
        """
        if not hasattr(self, "weights"):
            raise RuntimeError("Combiner not fitted. Call fit() first.")

        return sum(w * pred for w, pred in zip(self.weights, predictions))
