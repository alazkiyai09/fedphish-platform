"""Figure 5: Attack Success Rate Over Rounds"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate(output_dir: Path):
    """Generate Figure 5: Attack success rate over training rounds."""
    print("Generating Figure 5: Attack Success Rate Over Rounds")

    rounds = np.arange(0, 25)

    # Simulated attack success rates
    # Label flip attack
    label_flip_fedavg = [0] + list(0.25 * (1 - np.exp(-np.arange(1, 25) / 5)))
    label_flip_krum = [0] + list(0.12 * (1 - np.exp(-np.arange(1, 25) / 6)))
    label_flip_fedphish = [0] + list(0.03 * (1 - np.exp(-np.arange(1, 25) / 8)))

    # Backdoor attack
    backdoor_fedavg = [0] + list(0.35 * (1 - np.exp(-np.arange(1, 25) / 4)))
    backdoor_krum = [0] + list(0.08 * (1 - np.exp(-np.arange(1, 25) / 7)))
    backdoor_fedphish = [0] + list(0.02 * (1 - np.exp(-np.arange(1, 25) / 10)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Label Flip Attack
    ax1.plot(rounds, np.array(label_flip_fedavg) * 100, 'o-', label='FedAvg',
             color='#F59E0B', linewidth=2, markersize=5)
    ax1.plot(rounds, np.array(label_flip_krum) * 100, '^-', label='Krum',
             color='#10B981', linewidth=2, markersize=5)
    ax1.plot(rounds, np.array(label_flip_fedphish) * 100, 's-', label='FedPhish',
             color='#0072B2', linewidth=2.5, markersize=6)

    ax1.set_xlabel('Training Round', fontsize=11)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=11)
    ax1.set_title('(a) Label Flip Attack (20% Malicious)', fontsize=12, weight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 28)

    # Plot 2: Backdoor Attack
    ax2.plot(rounds, np.array(backdoor_fedavg) * 100, 'o-', label='FedAvg',
             color='#F59E0B', linewidth=2, markersize=5)
    ax2.plot(rounds, np.array(backdoor_krum) * 100, '^-', label='Krum',
             color='#10B981', linewidth=2, markersize=5)
    ax2.plot(rounds, np.array(backdoor_fedphish) * 100, 's-', label='FedPhish',
             color='#0072B2', linewidth=2.5, markersize=6)

    ax2.set_xlabel('Training Round', fontsize=11)
    ax2.set_ylabel('Backdoor Success Rate (%)', fontsize=11)
    ax2.set_title('(b) Backdoor Attack (20% Malicious)', fontsize=12, weight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 38)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_attacks.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_attacks.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Figure 5 saved to", output_dir)


if __name__ == "__main__":
    generate(Path("figures/output"))
