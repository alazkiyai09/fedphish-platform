"""Figures package for FedPhish paper."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5

# Color palette
COLORS = {
    'fedphish': '#0072B2',
    'fedavg': '#F59E0B',
    'local': '#10B981',
    'centralized': '#DC2626',
    'attack': '#EF4444',
    'defense': '#22C55E',
    'gray': '#6B7280',
}

# Create output directory
output_dir = Path("figures/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Import individual figure generators
from . import architecture, convergence, non_iid, privacy_pareto, attacks, fairness

print("✅ Figures package initialized")
