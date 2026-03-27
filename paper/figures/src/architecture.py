"""Figure 1: FedPhish System Architecture"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from pathlib import Path
import numpy as np


def generate(output_dir: Path):
    """Generate Figure 1: System Architecture diagram."""
    print("Generating Figure 1: FedPhish System Architecture")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    color_bank = '#0072B2'
    color_server = '#F59E0B'
    color_privacy = '#22C55E'
    color_security = '#DC2626'
    color_data = '#6B7280'

    # Title
    ax.text(6, 9.5, 'FedPhish Architecture', ha='center', va='top',
            fontsize=16, weight='bold')

    # ===== Banks (Clients) =====
    bank_y_positions = [7.5, 5.5, 3.5]
    bank_labels = ['Bank A', 'Bank B', 'Bank C']

    for i, (y, label) in enumerate(zip(bank_y_positions, bank_labels)):
        # Bank box
        bank_box = FancyBboxPatch((0.3, y - 0.4), 2.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color_bank, alpha=0.2,
                                   edgecolor=color_bank, linewidth=2)
        ax.add_patch(bank_box)
        ax.text(1.55, y, label, ha='center', va='center',
                fontsize=11, weight='bold')

        # Data component
        data_box = FancyBboxPatch((0.5, y - 0.25), 0.6, 0.5,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color_data, alpha=0.3)
        ax.add_patch(data_box)
        ax.text(0.8, y, 'Data', ha='center', va='center', fontsize=8)

        # Model component
        model_box = FancyBboxPatch((1.2, y - 0.25), 0.6, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color_bank, alpha=0.4)
        ax.add_patch(model_box)
        ax.text(1.5, y, 'Model', ha='center', va='center', fontsize=8)

        # Privacy components
        priv_box = FancyBboxPatch((1.9, y - 0.25), 0.8, 0.5,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color_privacy, alpha=0.3)
        ax.add_patch(priv_box)
        ax.text(2.3, y, 'DP+HE', ha='center', va='center', fontsize=7)

        # Security component
        sec_box = FancyBboxPatch((2.4, y - 0.35), 0.35, 0.7,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color_security, alpha=0.3)
        ax.add_patch(sec_box)
        ax.text(2.575, y, 'ZK', ha='center', va='center', fontsize=7)

    # Ellipsis for more banks
    ax.text(1.55, 2.2, '⋮', ha='center', fontsize=20, color=color_bank)
    ax.text(1.55, 1.5, 'Bank N', ha='center', va='center',
            fontsize=11, weight='bold', color=color_bank)

    # ===== Aggregation Server =====
    server_center_x = 6.5
    server_center_y = 5

    # Main server box
    server_box = FancyBboxPatch((4.5, 2.5), 4, 5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_server, alpha=0.15,
                                 edgecolor=color_server, linewidth=2.5)
    ax.add_patch(server_box)
    ax.text(6.5, 7.2, 'Aggregation Server', ha='center', va='center',
            fontsize=12, weight='bold', color=color_server)

    # TEE component
    tee_box = FancyBboxPatch((5, 5.5), 3, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor='#9333EA', alpha=0.25,
                              edgecolor='#9333EA', linewidth=2)
    ax.add_patch(tee_box)
    ax.text(6.5, 6.1, 'TEE (Gramine/SGX)', ha='center', va='center',
            fontsize=10, weight='bold', color='#9333EA')
    ax.text(6.5, 5.7, 'Secure Aggregation', ha='center', va='center',
            fontsize=8, color='#9333EA')

    # Verifier component
    verifier_box = FancyBboxPatch((5, 4), 3, 1,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color_security, alpha=0.2,
                                   edgecolor=color_security, linewidth=1.5)
    ax.add_patch(verifier_box)
    ax.text(6.5, 4.5, 'ZK Proof Verifier', ha='center', va='center',
            fontsize=9, weight='bold', color=color_security)

    # Defense component
    defense_box = FancyBboxPatch((5, 2.8), 3, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#22C55E', alpha=0.2,
                                  edgecolor='#22C55E', linewidth=1.5)
    ax.add_patch(defense_box)
    ax.text(6.5, 3.2, 'Byzantine Defense', ha='center', va='center',
            fontsize=9, weight='bold', color='#22C55E')

    # ===== Global Model =====
    global_model_box = FancyBboxPatch((9.5, 4), 2, 2,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color_bank, alpha=0.25,
                                       edgecolor=color_bank, linewidth=2)
    ax.add_patch(global_model_box)
    ax.text(10.5, 5, 'Global\nModel', ha='center', va='center',
            fontsize=11, weight='bold', color=color_bank)

    # ===== Arrows =====
    # Client to Server arrows
    for y in bank_y_positions:
        arrow = FancyArrowPatch((2.9, y), (4.4, server_center_y),
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle='->', mutation_scale=20,
                                linewidth=1.5, color='#333', alpha=0.6)
        ax.add_patch(arrow)

    # Server to Global Model arrow
    arrow = FancyArrowPatch((8.6, 5), (9.4, 5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color=color_server)
    ax.add_patch(arrow)

    # Global Model to Clients (dashed)
    for y in bank_y_positions:
        arrow = FancyArrowPatch((9.4, 5.3), (2.9, y + 0.3),
                                connectionstyle="arc3,rad=-0.3",
                                arrowstyle='->', mutation_scale=15,
                                linewidth=1.5, color=color_bank,
                                linestyle='--', alpha=0.5)
        ax.add_patch(arrow)

    # ===== Legend =====
    legend_y = 0.5
    legend_items = [
        (color_bank, 'Client/Model'),
        (color_server, 'Aggregation'),
        (color_privacy, 'Privacy (DP/HE)'),
        (color_security, 'Security (ZK)'),
        ('#9333EA', 'TEE'),
    ]

    for i, (color, label) in enumerate(legend_items):
        rect = Rectangle((0.5 + i * 2.2, legend_y), 0.3, 0.3,
                          facecolor=color, alpha=0.4, edgecolor=color)
        ax.add_patch(rect)
        ax.text(0.9 + i * 2.2, legend_y + 0.15, label, ha='left', va='center',
                fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Figure 1 saved to", output_dir)


if __name__ == "__main__":
    generate(Path("figures/output"))
