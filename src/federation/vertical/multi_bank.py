def simulate_multi_bank_split(banks: list[str]) -> dict:
    return {"banks": banks, "feature_partitions": {bank: index + 1 for index, bank in enumerate(banks)}}
