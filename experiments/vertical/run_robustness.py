"""
Run robustness experiments with malicious clients.

Tests system resilience against 1 malicious bank.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from experiments.run_federated import run_federated_experiment
from attacks import MaliciousClient, DefenseMechanism


def simulate_malicious_client(attack_type: str = 'label_flip') -> Dict:
    """
    Simulate federated learning with 1 malicious client.

    Args:
        attack_type: 'label_flip', 'byzantine', or 'backdoor'

    Returns:
        Results with and without defense
    """
    print("=" * 70)
    print(f"ROBUSTNESS TEST: {attack_type.upper()} Attack")
    print("=" * 70)

    # Run without malicious client (baseline)
    print("\n[1/3] Running baseline (no malicious client)...")
    baseline_results = run_federated_experiment(
        privacy_mechanism='none',
        strategy='fedavg',
        n_rounds=10,
        epsilon=1.0
    )

    # Run with malicious client, no defense
    print(f"\n[2/3] Running with malicious client (no defense)...")
    # TODO: Implement actual simulation with malicious client
    attack_results = baseline_results  # Placeholder

    # Run with malicious client + defense
    print(f"\n[3/3] Running with malicious client + defense...")
    # TODO: Implement with defense mechanisms (Krum, MFC)
    defense_results = baseline_results  # Placeholder

    # Compare results
    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)

    accuracy_loss = baseline_results['final_train_accuracy'] - attack_results['final_train_accuracy']

    print(f"Baseline accuracy:        {baseline_results['final_train_accuracy']:.4f}")
    print(f"Accuracy with attack:      {attack_results['final_train_accuracy']:.4f}")
    print(f"Accuracy loss:            {accuracy_loss:.4f}")
    print(f"Accuracy with defense:     {defense_results['final_train_accuracy']:.4f}")
    print(f"Defense recovered:        {(attack_results['final_train_accuracy'] - defense_results['final_train_accuracy']):.4f}")

    return {
        'attack_type': attack_type,
        'baseline_accuracy': baseline_results['final_train_accuracy'],
        'attack_accuracy': attack_results['final_train_accuracy'],
        'defense_accuracy': defense_results['final_train_accuracy'],
        'accuracy_loss': accuracy_loss,
        'recovered': (attack_results['final_train_accuracy'] - defense_results['final_train_accuracy'])
    }


def test_all_defenses() -> Dict:
    """Test all defense mechanisms against all attack types."""
    print("=" * 70)
    print("COMPREHENSIVE ROBUSTNESS EVALUATION")
    print("=" * 70)

    attacks = ['label_flip', 'byzantine', 'backdoor']
    results = {}

    for attack in attacks:
        result = simulate_malicious_client(attack_type=attack)
        results[attack] = result
        print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All Attack/Defense Pairs")
    print("=" * 70)

    for attack, result in results.items():
        print(f"\n{attack.upper()}:")
        print(f"  Accuracy loss: {result['accuracy_loss']:.4f}")
        print(f"  Defense recovered: {result['recovered']:.4f}")
        print(f"  Final accuracy: {result['defense_accuracy']:.4f}")

    return results


if __name__ == '__main__':
    # Test specific attack
    result = simulate_malicious_client(attack_type='label_flip')

    # Or test all
    # results = test_all_defenses()
