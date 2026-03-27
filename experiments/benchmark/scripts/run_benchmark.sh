#!/bin/bash
# Run full FedPhish Benchmark Suite

set -e

echo "======================================"
echo "FedPhish Benchmark Suite"
echo "======================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p data/raw data/processed data/cache results/runs results/tables results/figures

# Run the benchmark
echo ""
echo "Running benchmark..."
echo ""

python -m src.experiments.benchmark

echo ""
echo "======================================"
echo "Benchmark completed!"
echo "Results saved to: ./results/"
echo "======================================"
