def inspect_gradients(gradients: list[list[float]]) -> dict:
    return {"clients": len(gradients), "dimensions": len(gradients[0]) if gradients else 0}
