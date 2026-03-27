"""Benchmark all privacy levels and configurations."""
import argparse
import logging
import yaml
from run_federated import run_federated, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    base_config = load_config(args.config)

    results = {}
    for privacy_level in [1, 2, 3]:
        logger.info(f"\n{'='*60}\nBenchmarking Privacy Level {privacy_level}\n{'='*60}")
        config = base_config.copy()
        config["experiment"]["privacy_level"] = privacy_level

        try:
            result = run_federated(config)
            results[f"level_{privacy_level}"] = result
        except Exception as e:
            logger.error(f"Error in privacy level {privacy_level}: {e}")
            results[f"level_{privacy_level}"] = {"error": str(e)}

    # Save results
    with open("results/benchmark_results.yaml", "w") as f:
        yaml.dump(results, f)

    logger.info("\nBenchmark complete!")
    for level, result in results.items():
        if "final_accuracy" in result:
            print(f"{level}: Accuracy = {result['final_accuracy']:.4f}")

if __name__ == "__main__":
    main()
