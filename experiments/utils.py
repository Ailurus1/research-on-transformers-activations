from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from datasets import Dataset


@dataclass
class SampleConfig:
    split: str
    max_samples: int
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataset(dataset: Dataset, config: SampleConfig) -> Dataset:
    sample_size = min(config.max_samples, len(dataset))
    if sample_size <= 0:
        return dataset.select([])
    shuffled = dataset.shuffle(seed=config.seed)
    return shuffled.select(range(sample_size))


def chunked(items: List, batch_size: int) -> Iterable[List]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]
