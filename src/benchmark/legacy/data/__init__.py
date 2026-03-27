"""Data loading, partitioning, and attack injection."""

from .loader import PhishingDataLoader, load_phishing_dataset
from .partitioner import DataPartitioner, partition_iid, partition_non_iid, partition_label_skew
from .attack_injector import AttackInjector, inject_label_flip, inject_backdoor

__all__ = [
    "PhishingDataLoader",
    "load_phishing_dataset",
    "DataPartitioner",
    "partition_iid",
    "partition_non_iid",
    "partition_label_skew",
    "AttackInjector",
    "inject_label_flip",
    "inject_backdoor",
]
