"""
Unit tests for DistilBERT+LoRA model.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from models.distilbert_lora import DistilBertLoRAForPhishing, LoRALayer


class TestLoRALayer:
    """Test LoRA layer implementation."""

    def test_lora_init(self):
        """Test LoRA layer initialization."""
        lora = LoRALayer(in_dim=768, out_dim=768, rank=4, alpha=1.0)

        assert lora.in_dim == 768
        assert lora.out_dim == 768
        assert lora.rank == 4

        # Check parameters exist
        assert lora.lora_A.shape == (768, 4)
        assert lora.lora_B.shape == (4, 768)

    def test_lora_forward(self):
        """Test LoRA forward pass."""
        lora = LoRALayer(in_dim=768, out_dim=768, rank=4)
        x = torch.randn(2, 768)

        output = lora(x)

        assert output.shape == (2, 768)

    def test_lora_parameters_trainable(self):
        """Test that LoRA parameters are trainable."""
        lora = LoRALayer(in_dim=768, out_dim=768, rank=4)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


class TestDistilBertLoRA:
    """Test DistilBERT with LoRA model."""

    def test_model_init(self):
        """Test model initialization."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        assert model.num_labels == 2
        assert model.lora_rank == 4
        assert len(model.lora_layers) == 6

    def test_model_forward(self):
        """Test forward pass."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        batch_size = 4
        seq_len = 128

        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        logits = model(input_ids, attention_mask)

        assert logits.shape == (batch_size, 2)

    def test_base_model_frozen(self):
        """Test that base DistilBERT is frozen."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        for param in model.distilbert.parameters():
            assert not param.requires_grad

    def test_lora_params_trainable(self):
        """Test that LoRA params are trainable."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        lora_params = model.get_lora_params()

        assert len(lora_params) == 12  # 6 layers * 2 matrices (A and B)
        assert all(p.requires_grad for p in lora_params)

    def test_get_and_set_lora_params(self):
        """Test getting and setting LoRA parameters."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        # Get parameters
        params = model.get_lora_params()

        # Convert to numpy and back (simulates FL communication)
        params_np = [p.detach().cpu().numpy() for p in params]
        params_restored = [torch.from_numpy(p) for p in params_np]

        # Set parameters
        model.set_lora_params(params_restored)

        # Verify they match
        new_params = model.get_lora_params()
        for orig, new in zip(params, new_params):
            assert torch.allclose(orig, new)

    def test_classifier_params(self):
        """Test classifier parameter handling."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        classifier_params = model.get_classifier_params()

        assert len(classifier_params) == 2  # weight and bias
        assert classifier_params[0].shape == (2, 768)  # weight
        assert classifier_params[1].shape == (2,)  # bias

    def test_get_trainable_params(self):
        """Test getting all trainable params."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        trainable_params = model.get_trainable_params()

        assert len(trainable_params) == 14  # 12 LoRA + 2 classifier

    def test_set_trainable_params(self):
        """Test setting all trainable params."""
        model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)

        lora_params = model.get_lora_params()
        classifier_params = model.get_classifier_params()

        # Convert to numpy and back
        lora_np = [torch.from_numpy(p.detach().cpu().numpy()) for p in lora_params]
        classifier_np = [torch.from_numpy(p.detach().cpu().numpy()) for p in classifier_params]

        model.set_trainable_params(lora_np, classifier_np)

        # Verify
        new_lora = model.get_lora_params()
        new_classifier = model.get_classifier_params()

        for orig, new in zip(lora_params + classifier_params, new_lora + new_classifier):
            assert torch.allclose(orig, new)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
