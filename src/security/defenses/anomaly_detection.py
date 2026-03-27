def anomaly_score(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return max(abs(value - mean) for value in values)
