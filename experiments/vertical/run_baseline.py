"""
Run baseline experiments: centralized training (all data pooled).

This serves as the upper bound for accuracy (no privacy constraints).
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from models import DistilBertLoRAForPhishing
from data import get_tokenizer
from banks import GlobalBank, RegionalBank, DigitalBank, CreditUnion, InvestmentBank
from evaluation import compute_metrics


def run_baseline(n_epochs: int = 5,
                learning_rate: float = 0.001,
                batch_size: int = 32) -> Dict:
    """
    Run centralized training with all data pooled.

    Args:
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("BASELINE: Centralized Training (All Data Pooled)")
    print("=" * 70)
    print()

    # Create all banks
    print("Loading data from all 5 banks...")
    banks = [
        GlobalBank(data_path='data/bank_datasets'),
        RegionalBank(data_path='data/bank_datasets'),
        DigitalBank(data_path='data/bank_datasets'),
        CreditUnion(data_path='data/bank_datasets'),
        InvestmentBank(data_path='data/bank_datasets')
    ]

    # Load all data
    all_emails = []
    all_labels = []

    for bank in banks:
        bank.load_data('train')

        for batch in bank.train_loader:
            all_emails.append(batch['input_ids'])
            all_labels.append(batch['labels'])

    # Concatenate all data
    all_input_ids = torch.cat(all_emails, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Total samples: {len(all_labels)}")

    # Create model
    print("\nInitializing DistilBERT+LoRA model...")
    model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Training
    print(f"\nTraining on device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print()

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=learning_rate)
    loss_fct = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels_epoch = []

        # Process in batches
        batch_size = 32
        n_samples = len(all_labels)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            input_ids = all_input_ids[i:end_idx].to(device)
            # Need to recreate attention_mask
            attention_mask = (input_ids != 0).long().to(device)
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

            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels_epoch.extend(labels.detach().cpu().numpy())

        # Metrics
        avg_loss = total_loss / (n_samples / batch_size)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels_epoch))

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Evaluate on each bank's test set
    print("\nEvaluating on each bank's test set...")
    per_bank_metrics = []

    for bank in banks:
        bank.load_data('test')
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in bank.test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds)
        )

        per_bank_metrics.append({
            'bank': bank.profile.name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

    print("\nPer-Bank Results:")
    print("-" * 70)
    for metrics in per_bank_metrics:
        print(f"{metrics['bank']:<20} Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1']:.4f}")

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
        n_epochs=5,
        learning_rate=0.001,
        batch_size=32
    )
