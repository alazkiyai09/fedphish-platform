def label_flip_attack(labels: list[int], ratio: float = 0.2) -> list[int]:
    flips = int(len(labels) * ratio)
    return [1 - label if index < flips else label for index, label in enumerate(labels)]
