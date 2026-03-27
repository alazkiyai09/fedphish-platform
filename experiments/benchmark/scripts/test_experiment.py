#!/usr/bin/env python3
"""Quick test script for a minimal experiment."""

from omegaconf import OmegaConf
import sys
sys.path.insert(0, '.')

from src.experiments.runner import ExperimentRunner

# Load and merge configs
config = OmegaConf.load('config/benchmark.yaml')
config = OmegaConf.merge(
    config,
    OmegaConf.load('config/dataset/phishing.yaml'),
    OmegaConf.load('config/model/classical_ml.yaml'),
    OmegaConf.load('config/federation/fedavg.yaml'),
)

# Override for quick test
config.benchmark.num_runs = 1
config.federation.server.num_rounds = 3  # Reduce rounds for quick test

runner = ExperimentRunner(config)

# Run one experiment
result = runner.run(
    model_type='xgboost',
    federation_type='fedavg',
    data_distribution='iid',
    attack_type='none',
    privacy_mechanism='none',
    run_id=0,
)

print('Experiment completed successfully!')
print(f'Accuracy: {result.metrics.get("accuracy", "N/A")}')
