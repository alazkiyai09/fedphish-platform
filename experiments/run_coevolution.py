"""Runnable coevolution experiment for FedPhish security research."""

from __future__ import annotations

from src.security.coevolution.arms_race import simulate_arms_race
from src.security.coevolution.game_theory import compute_equilibrium
from src.security.coevolution.scenarios import list_scenarios


def run(rounds: int = 10) -> dict[str, object]:
    return {
        "experiment": "coevolution",
        "rounds": rounds,
        "scenarios": list_scenarios(),
        "arms_race": simulate_arms_race(rounds),
        "equilibrium": compute_equilibrium(),
    }
