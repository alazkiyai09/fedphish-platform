"""Evaluate privacy-utility tradeoffs."""
import argparse
import logging
import yaml
import numpy as np
from run_federated import run_federated, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 1.0, 5.0, 10.0])
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    base_config = load_config(args.config)

    results = {}
    for epsilon in args.epsilons:
        logger.info(f"\nEvaluating ε={epsilon}")
        config = base_config.copy()
        config["privacy"]["epsilon"] = epsilon

        try:
            result = run_federated(config)
            results[f"epsilon_{epsilon}"] = {
                "accuracy": result.get("final_accuracy", 0.0),
                "epsilon": epsilon,
            }
        except Exception as e:
            logger.error(f"Error at ε={epsilon}: {e}")
            results[f"epsilon_{epsilon}"] = {"error": str(e), "epsilon": epsilon}

    with open("results/privacy_eval_results.yaml", "w") as f:
        yaml.dump(results, f)

    logger.info("Privacy evaluation complete!")

if __name__ == "__main__":
    main()
