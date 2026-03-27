"""
Run local-only training experiments.

Each bank trains independently without federation.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from models import DistilBertLoRAForPhishing
from banks import GlobalBank, RegionalBank, DigitalBank, CreditUnion, InvestmentBank
from evaluation import compute_metrics


def run_local_training(bank_name: str,
                      n_epochs: int = 5,
                      learning_rate: float = 0.001) -> Dict:
    """
    Train a model on a single bank's data (no federation).

    Args:
        bank_name: Name of bank ('global_bank', 'regional_bank', etc.)
        n_epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 70}")
    print(f"LOCAL TRAINING: {bank_name}")
    print(f"{'=' * 70}")

    # Create bank
    if bank_name == 'global_bank':
        bank = GlobalBank(data_path='data/bank_datasets')
    elif bank_name == 'regional_bank':
        bank = RegionalBank(data_path='data/bank_datasets')
    elif bank_name == 'digital_bank':
        bank = DigitalBank(data_path='data/bank_datasets')
    elif bank_name == 'credit_union':
        bank = CreditUnion(data_path='data/bank_datasets')
    elif bank_name == 'investment_bank':
        bank = InvestmentBank(data_path='data/bank_datasets')
    else:
        raise ValueError(f"Unknown bank: {bank_name}")

    # Load data
    bank.load_data('train')
    bank.load_data('test')

    print(f"Training samples: {len(bank.train_loader.dataset)}")
    print(f"Test samples: {len(bank.test_loader.dataset)}")
    print(f"Data quality: {bank.get_data_quality()}")

    # Create model
    model = DistilBertLoRAForPhishing(num_labels=2, lora_rank=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Training
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=learning_rate)
    loss_fct = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in bank.train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(bank.train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []

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

    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test F1: {metrics['f1']:.4f}")

    return {
        'bank': bank_name,
        'final_loss': train_losses[-1],
        'final_accuracy': train_accuracies[-1],
        'test_accuracy': metrics['accuracy'],
        'test_f1': metrics['f1'],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }


def run_all_local_banks() -> Dict:
    """Run local training for all banks individually."""
    print("=" * 70)
    print("LOCAL-ONLY TRAINING (No Federation)")
    print("=" * 70)

    banks = [
        'global_bank',
        'regional_bank',
        'digital_bank',
        'credit_union',
        'investment_bank'
    ]

    results = {}

    for bank_name in banks:
        result = run_local_training(
            bank_name=bank_name,
            n_epochs=5,
            learning_rate=0.001
        )
        results[bank_name] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Local Training Results")
    print("=" * 70)
    print(f"{'Bank':<20} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 70)

    for bank_name, result in results.items():
        print(f"{bank_name:<20} {result['final_accuracy']:<12.4f} {result['test_accuracy']:<12.4f}")

    # Average across banks
    avg_accuracy = np.mean([r['final_accuracy'] for r in results.values()])
    print("-" * 70)
    print(f"{'Average':<20} {avg_accuracy:<12.4f}")

    return results


if __name__ == '__main__':
    results = run_all_local_banks()
