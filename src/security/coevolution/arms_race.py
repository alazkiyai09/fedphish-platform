def simulate_arms_race(rounds: int) -> dict:
    return {"rounds": rounds, "winner": "defense" if rounds % 2 else "attack"}
