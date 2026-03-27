"""
Run baseline experiments: centralized training (all data pooled).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import directly from modules to avoid relative import issues
from models.distilbert_lora import DistilBertLoRAForPhishing
from data.phishing_dataset import PhishingEmailDataset, get_tokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from evaluation.metrics import compute_metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def create_dummy_phishing_data(n_samples: int, seq_length: int = 128):
    """Create dummy phishing dataset for demonstration."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Create random input_ids (simulating tokenized text)
    input_ids = torch.randint(0, 30000, (n_samples, seq_length))

    # Create attention mask
    attention_mask = (input_ids != 0).long()

    # Create labels (0: safe, 1: phishing)
    # Create imbalanced dataset (20% phishing)
    labels = torch.zeros(n_samples, dtype=torch.long)
    phishing_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    labels[phishing_indices] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def run_baseline(n_epochs: int = 3,
                learning_rate: float = 0.001,
                batch_size: int = 32) -> dict:
    """
    Run centralized training with all data pooled.
    """
    print("=" * 70)
    print("BASELINE: Centralized Training (All Data Pooled)")
    print("=" * 70)
    print()

    # Simulate 5 banks with different data sizes
    print("Loading data from all 5 banks...")
    bank_configs = [
        {'name': 'Global Bank', 'samples': 10000, 'quality': 0.95},
        {'name': 'Regional Bank', 'samples': 3000, 'quality': 0.82},
        {'name': 'Digital Bank', 'samples': 5000, 'quality': 0.88},
        {'name': 'Credit Union', 'samples': 1500, 'quality': 0.75},
        {'name': 'Investment Bank', 'samples': 1000, 'quality': 0.92},
    ]

    # Create dummy data for each bank
    all_data = []
    for config in bank_configs:
        data = create_dummy_phishing_data(config['samples'])
        all_data.append(data)
        print(f"  {config['name']}: {config['samples']} samples (quality: {config['quality']})")

    # Pool all data
    all_input_ids = torch.cat([d['input_ids'] for d in all_data], dim=0)
    all_attention_mask = torch.cat([d['attention_mask'] for d in all_data], dim=0)
    all_labels = torch.cat([d['labels'] for d in all_data], dim=0)

    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Phishing ratio: {all_labels.float().mean():.2%}")

    # Create model
    print("\nInitializing DistilBERT+LoRA model...")
    model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training
    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print()

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels_epoch = []

        n_samples = len(all_labels)
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            input_ids = all_input_ids[i:end_idx].to(device)
            attention_mask = all_attention_mask[i:end_idx].to(device)
            labels = all_labels[i:end_idx].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Compute loss
            loss = loss_fct(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels_epoch.extend(labels.detach().cpu().numpy())

        # Metrics
        avg_loss = total_loss / n_batches
        accuracy = accuracy_score(all_labels_epoch, all_preds)

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Evaluate on test data (20% of each bank's data)
    print("\nEvaluating on test sets...")
    per_bank_metrics = []

    model.eval()

    for idx, config in enumerate(bank_configs):
        # Create test data
        test_samples = int(config['samples'] * 0.2)
        test_data = create_dummy_phishing_data(test_samples)

        all_preds = []
        all_labels_test = []

        with torch.no_grad():
            input_ids = test_data['input_ids'].to(device)
            attention_mask = test_data['attention_mask'].to(device)
            labels = test_data['labels']

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.numpy())

        accuracy = accuracy_score(all_labels_test, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_test, all_preds, average='binary', zero_division=0
        )

        per_bank_metrics.append({
            'bank': config['name'],
            'samples': test_samples,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    print("\nPer-Bank Test Results:")
    print("-" * 70)
    for metrics in per_bank_metrics:
        print(f"{metrics['bank']:<20} Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    # Overall results
    final_accuracy = train_accuracies[-1]
    final_loss = train_losses[-1]

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"Final Training Accuracy: {final_accuracy:.4f}")
    print(f"Final Training Loss: {final_loss:.4f}")

    return {
        'approach': 'centralized',
        'n_epochs': n_epochs,
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'per_bank_metrics': per_bank_metrics,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }


if __name__ == '__main__':
    results = run_baseline(
        n_epochs=3,  # Using 3 epochs for faster demonstration
        learning_rate=0.001,
        batch_size=32
    )
