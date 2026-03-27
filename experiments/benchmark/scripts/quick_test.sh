#!/bin/bash
# Quick smoke test for the benchmark

set -e

echo "Running quick smoke test..."

# Run unit tests
echo "Running unit tests..."
pytest tests/ -v --tb=short

# Run a minimal experiment
echo ""
echo "Running minimal experiment..."
python3 scripts/test_experiment.py

echo ""
echo "Smoke test passed!"
