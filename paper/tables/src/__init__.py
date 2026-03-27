"""Tables package for FedPhish paper."""

import numpy as np
import pandas as pd
from pathlib import Path

# Import individual table generators
from . import detection_performance, privacy_utility, robustness, overhead

# Create output directory
output_dir = Path("tables/output")
output_dir.mkdir(parents=True, exist_ok=True)

print("✅ Tables package initialized")
