"""
Model explanation for phishing detection.

Provides interpretability via SHAP values and attention visualization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    logger.warning("SHAP not available. Install with: pip install shap")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class SHAPExplainer:
    """
    Explain predictions using SHAP (SHapley Additive exPlanations).

    Works with both transformer and XGBoost components.
    """

    def __init__(
        self,
        model,
        model_type: str = "xgboost",  # 'xgboost' or 'transformer'
        tokenizer=None,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Model to explain
            model_type: Type of model ('xgboost' or 'transformer')
            tokenizer: Tokenizer (for transformer models)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")

        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer

        # Initialize explainer based on model type
        if model_type == "xgboost":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "transformer":
            self.explainer = shap.DeepExplainer(model, None)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized SHAP explainer for {model_type}")

    def explain_instance(
        self,
        instance: Any,
        num_features: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explain a single prediction.

        Args:
            instance: Input instance (features or text)
            num_features: Number of top features to show

        Returns:
            Tuple of (feature_values, shap_values)
        """
        shap_values = self.explainer.shap_values(instance)

        if self.model_type == "transformer" and isinstance(shap_values, list):
            shap_values = shap_values[0]  # For binary classification

        # Get absolute values for ranking
        if len(shap_values.shape) > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)

        # Get top features
        top_indices = np.argsort(importance)[-num_features:][::-1]

        return top_indices, shap_values

    def explain_batch(
        self,
        instances: Any,
    ) -> np.ndarray:
        """
        Explain multiple instances.

        Args:
            instances: Batch of instances

        Returns:
            SHAP values for all instances
        """
        return self.explainer.shap_values(instances)

    def plot_waterfall(
        self,
        instance: Any,
        save_path: Optional[str] = None,
    ):
        """
        Create waterfall plot for single instance.

        Args:
            instance: Input instance
            save_path: Path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        shap_values = self.explainer.shap_values(instance)

        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
            )
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved waterfall plot to {save_path}")

        plt.close()

    def plot_summary(
        self,
        instances: Any,
        save_path: Optional[str] = None,
    ):
        """
        Create summary plot for multiple instances.

        Args:
            instances: Batch of instances
            save_path: Path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        shap_values = self.explainer.shap_values(instances)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, instances, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved summary plot to {save_path}")

        plt.close()


class AttentionVisualizer:
    """
    Visualize attention weights from transformer models.

    Shows which tokens the model focuses on for predictions.
    """

    def __init__(
        self,
        model,
        tokenizer,
    ):
        """
        Initialize attention visualizer.

        Args:
            model: Transformer model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

        logger.info("Initialized attention visualizer")

    def get_attention_weights(
        self,
        text: str,
        layer: int = 0,
        head: int = 0,
    ) -> np.ndarray:
        """
        Get attention weights for specific layer and head.

        Args:
            text: Input text
            layer: Layer index
            head: Attention head index

        Returns:
            Attention weight matrix
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Get attention outputs
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
            )

        # Extract attention for specific layer and head
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0, head].cpu().numpy()

        return attention

    def visualize_attention(
        self,
        text: str,
        layer: int = 0,
        head: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Visualize attention as heatmap.

        Args:
            text: Input text
            layer: Layer index
            head: Attention head index
            save_path: Path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        # Get attention weights
        attention = self.get_attention_weights(text, layer, head)

        # Get tokens
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:attention.shape[0]]  # Truncate to match

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"Attention Weights - Layer {layer}, Head {head}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attention visualization to {save_path}")

        plt.close()

    def visualize_all_heads(
        self,
        text: str,
        layer: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Visualize all attention heads in a layer.

        Args:
            text: Input text
            layer: Layer index
            save_path: Path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Get attention outputs
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
            )

        # Extract attention for layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0].cpu().numpy()

        # Get tokens
        tokens = self.tokenizer.tokenize(text)

        # Number of heads
        num_heads = attention.shape[0]

        # Create subplots
        cols = 4
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head in range(num_heads):
            ax = axes[head]
            sns.heatmap(
                attention[head],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
                cbar=True,
                ax=ax,
            )
            ax.set_title(f"Head {head}")

        # Hide extra subplots
        for head in range(num_heads, len(axes)):
            axes[head].axis("off")

        plt.suptitle(f"All Attention Heads - Layer {layer}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved all heads visualization to {save_path}")

        plt.close()


class ReportGenerator:
    """
    Generate explanation reports for predictions.

    Combines SHAP values, attention, and feature importance.
    """

    def __init__(
        self,
        shap_explainer: Optional[SHAPExplainer] = None,
        attention_visualizer: Optional[AttentionVisualizer] = None,
    ):
        """
        Initialize report generator.

        Args:
            shap_explainer: SHAP explainer instance
            attention_visualizer: Attention visualizer instance
        """
        self.shap_explainer = shap_explainer
        self.attention_visualizer = attention_visualizer

    def generate_report(
        self,
        text: str,
        prediction: int,
        probability: float,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate explanation report.

        Args:
            text: Input text
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            features: Engineered features (optional)
            feature_names: Feature names (optional)

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PHISHING DETECTION EXPLANATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Prediction summary
        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        lines.append(f"Prediction: {label}")
        lines.append(f"Confidence: {probability:.2%}")
        lines.append("")

        # Input text
        lines.append("Input Text:")
        lines.append("-" * 40)
        lines.append(text[:200] + "..." if len(text) > 200 else text)
        lines.append("")

        # SHAP explanations
        if self.shap_explainer is not None:
            lines.append("Feature Importance (SHAP):")
            lines.append("-" * 40)

            if self.shap_explainer.model_type == "xgboost" and features is not None:
                # Explain with feature importance
                shap_values = self.shap_explainer.explainer.shap_values(features.reshape(1, -1))

                if feature_names is not None:
                    importance = np.abs(shap_values[0])
                    top_indices = np.argsort(importance)[-5:][::-1]

                    for idx in top_indices:
                        value = shap_values[0][idx]
                        direction = "increases" if value > 0 else "decreases"
                        lines.append(
                            f"  {feature_names[idx]}: {direction} risk "
                            f"(SHAP value: {value:.4f})"
                        )
            else:
                lines.append("  (SHAP explanation available for feature-based models)")

            lines.append("")

        # Attention visualization mention
        if self.attention_visualizer is not None:
            lines.append("Model Attention:")
            lines.append("-" * 40)
            lines.append("  The model focuses on specific words and phrases to make predictions.")
            lines.append("  Use AttentionVisualizer.visualize_attention() to see attention weights.")
            lines.append("")

        # Recommendation
        lines.append("Recommendation:")
        lines.append("-" * 40)
        if prediction == 1:
            lines.append("  ⚠️  This appears to be a PHISHING attempt.")
            lines.append("  Recommended actions:")
            lines.append("    - Do not click any links")
            lines.append("    - Do not download attachments")
            lines.append("    - Report to security team")
        else:
            lines.append("  ✓ This appears to be LEGITIMATE.")
            lines.append("  Still recommended to:")
            lines.append("    - Verify sender identity")
            lines.append("    - Check for suspicious URLs")
            lines.append("    - Be cautious with requests for sensitive info")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(
        self,
        text: str,
        prediction: int,
        probability: float,
        output_path: str,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Generate and save report to file.

        Args:
            text: Input text
            prediction: Model prediction
            probability: Prediction probability
            output_path: Path to save report
            features: Engineered features
            feature_names: Feature names
        """
        report = self.generate_report(
            text=text,
            prediction=prediction,
            probability=probability,
            features=features,
            feature_names=feature_names,
        )

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Saved explanation report to {output_path}")


def create_explanation_report(
    text: str,
    prediction: int,
    probability: float,
    model_type: str = "xgboost",
    model=None,
    tokenizer=None,
    features: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Convenience function to create explanation report.

    Args:
        text: Input text
        prediction: Model prediction
        probability: Prediction probability
        model_type: Type of model
        model: Model instance
        tokenizer: Tokenizer (for transformers)
        features: Engineered features
        feature_names: Feature names
        output_path: Path to save report

    Returns:
        Report string
    """
    # Initialize explainers
    shap_explainer = None
    attention_visualizer = None

    if model is not None:
        if model_type == "xgboost":
            shap_explainer = SHAPExplainer(model, model_type="xgboost")
        elif model_type == "transformer" and tokenizer is not None:
            shap_explainer = SHAPExplainer(model, model_type="transformer", tokenizer=tokenizer)
            attention_visualizer = AttentionVisualizer(model, tokenizer)

    # Generate report
    generator = ReportGenerator(shap_explainer, attention_visualizer)
    report = generator.generate_report(
        text=text,
        prediction=prediction,
        probability=probability,
        features=features,
        feature_names=feature_names,
    )

    if output_path:
        generator.save_report(
            text=text,
            prediction=prediction,
            probability=probability,
            output_path=output_path,
            features=features,
            feature_names=feature_names,
        )

    return report
