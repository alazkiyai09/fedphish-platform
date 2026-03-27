"""
Transformer-based phishing detection model.

Uses DistilBERT with LoRA adapters for efficient fine-tuning.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
)

logger = logging.getLogger(__name__)


class DistilBERTPhishing(nn.Module):
    """
    DistilBERT model for phishing detection.

    Pre-trained DistilBERT with classification head.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Initialize DistilBERT model.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            dropout: Dropout probability
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout = dropout
        self.freeze_base = freeze_base

        # Load pre-trained model
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.config = self.distilbert.config

        # Classification head
        self.pre_classifier = nn.Linear(
            self.config.dim,
            self.config.dim,
        )
        self.classifier = nn.Linear(
            self.config.dim,
            num_labels,
        )
        self.dropout_layer = nn.Dropout(dropout)

        # Freeze base if requested
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            logger.info("Frozen DistilBERT base model")

        logger.info(f"Initialized DistilBERT model: {model_name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Logits
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        hidden_state = outputs.last_hidden_state[:, 0, :]

        # Classification head
        hidden_state = self.pre_classifier(hidden_state)
        hidden_state = nn.ReLU()(hidden_state)
        hidden_state = self.dropout_layer(hidden_state)
        logits = self.classifier(hidden_state)

        return logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract embeddings from [CLS] token.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Embeddings
        """
        with torch.no_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings


class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) adapter for efficient fine-tuning.

    Adds trainable low-rank decomposition to weight matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRA adapter.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of low-rank decomposition
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        # Reset B to zero (start with no adaptation)
        nn.init.zeros_(self.lora_B)

        logger.debug(
            f"Initialized LoRA adapter: rank={rank}, alpha={alpha}, "
            f"in={in_features}, out={out_features}"
        )

    def forward(
        self,
        x: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor
            original_weight: Original weight matrix to adapt

        Returns:
            Adapted output
        """
        # Original linear transformation
        result = torch.nn.functional.linear(x, original_weight)

        # LoRA adaptation: W + (A @ B) * scaling
        lora_result = torch.nn.functional.linear(
            self.lora_dropout(x),
            self.lora_B.T,
        )
        lora_result = torch.nn.functional.linear(
            lora_result,
            self.lora_A.T,
        ) * self.scaling

        return result + lora_result


class DistilBERTWithLoRA(nn.Module):
    """
    DistilBERT with LoRA adapters.

    Efficient fine-tuning by adding LoRA to attention layers.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        freeze_base: bool = True,
    ):
        """
        Initialize model with LoRA.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            freeze_base: Whether to freeze base model
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Load base model
        self.base_model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.config = self.base_model.config

        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.distilbert.parameters():
                param.requires_grad = False
            logger.info("Frozen DistilBERT base model")

        # Add LoRA adapters to query/key/value projections
        self.lora_layers = nn.ModuleDict()
        self._add_lora_to_attention()

        logger.info(
            f"Initialized DistilBERT with LoRA: "
            f"rank={lora_rank}, alpha={lora_alpha}"
        )

    def _add_lora_to_attention(self):
        """Add LoRA adapters to attention layers."""
        # Add LoRA to each layer's attention
        for i, layer in enumerate(self.base_model.distilbert.transformer.layer):
            # Query projection
            q_in = layer.attention.q_lin.in_features
            q_out = layer.attention.q_lin.out_features
            self.lora_layers[f"layer_{i}_q"] = LoRAAdapter(
                q_in, q_out, self.lora_rank, self.lora_alpha
            )

            # Key projection
            k_in = layer.attention.k_lin.in_features
            k_out = layer.attention.k_lin.out_features
            self.lora_layers[f"layer_{i}_k"] = LoRAAdapter(
                k_in, k_out, self.lora_rank, self.lora_alpha
            )

            # Value projection
            v_in = layer.attention.v_lin.in_features
            v_out = layer.attention.v_lin.out_features
            self.lora_layers[f"layer_{i}_v"] = LoRAAdapter(
                v_in, v_out, self.lora_rank, self.lora_alpha
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Logits
        """
        # Get embeddings
        inputs_embeds = self.base_model.distilbert.embeddings(input_ids)

        # Apply transformer layers with LoRA
        hidden_state = inputs_embeds
        for i, layer in enumerate(self.base_model.distilbert.transformer.layer):
            # Apply attention with LoRA
            query = self.lora_layers[f"layer_{i}_q"](
                hidden_state, layer.attention.q_lin.weight
            )
            key = self.lora_layers[f"layer_{i}_k"](
                hidden_state, layer.attention.k_lin.weight
            )
            value = self.lora_layers[f"layer_{i}_v"](
                hidden_state, layer.attention.v_lin.weight
            )

            # Rest of attention mechanism
            attn_output = layer.attention(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                query=query,
                key=key,
                value=value,
            )[0]

            # Apply FFN
            hidden_state = layer(attn_output, attention_mask)[0]

        # Pooling
        pooled = hidden_state[:, 0, :]
        pooled = self.base_model.pre_classifier(pooled)
        pooled = nn.ReLU()(pooled)
        pooled = nn.Dropout(0.1)(pooled)

        # Classification
        logits = self.base_model.classifier(pooled)

        return logits

    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable LoRA parameters."""
        params = []
        for layer in self.lora_layers.values():
            params.extend([layer.lora_A, layer.lora_B])
        return params

    def get_classifier_parameters(self) -> List[torch.nn.Parameter]:
        """Get classifier parameters."""
        return list(self.base_model.classifier.parameters())


class FeatureExtractor:
    """
    Extract features from DistilBERT for downstream tasks.

    Extracts embeddings from intermediate layers.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        layers: List[int] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: HuggingFace model name
            layers: Layers to extract from (None = last layer)
        """
        self.model_name = model_name
        self.layers = layers or [-1]  # Default to last layer

        # Load model
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Initialized feature extractor from {model_name}")

    def extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Extract embeddings from texts.

        Args:
            texts: List of texts
            batch_size: Batch size
            max_length: Maximum sequence length

        Returns:
            Embeddings array
        """
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                # Get outputs
                outputs = self.model(**inputs)

                # Extract [CLS] token from last hidden state
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def extract_with_attention(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and attention weights.

        Args:
            texts: List of texts
            batch_size: Batch size
            max_length: Maximum sequence length

        Returns:
            Tuple of (embeddings, attention_weights)
        """
        self.model.eval()
        all_embeddings = []
        all_attentions = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                # Get outputs with attention
                outputs = self.model(**inputs, output_attentions=True)

                # Extract embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

                # Extract attention from first layer, first head
                attention = outputs.attentions[0][:, 0, :, :].cpu().numpy()
                all_attentions.append(attention)

        embeddings = np.vstack(all_embeddings)
        attentions = np.vstack(all_attentions)

        return embeddings, attentions
