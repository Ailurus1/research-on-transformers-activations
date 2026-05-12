from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import Dataset


@dataclass
class SampleConfig:
    split: str
    max_samples: Optional[int] = None
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataset(dataset: Dataset, config: SampleConfig) -> Dataset:
    if config.max_samples is None:
        return dataset
    sample_size = min(config.max_samples, len(dataset))
    if sample_size <= 0:
        return dataset.select([])
    shuffled = dataset.shuffle(seed=config.seed)
    return shuffled.select(range(sample_size))


def chunked(items: List, batch_size: int) -> Iterable[List]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def _norm_ic_label(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()


def _match_class_description_to_model_id(class_str: str, ilsvrc: Dict[int, str]) -> int:
    n_full = _norm_ic_label(class_str)
    n_key = _norm_ic_label(class_str.split(",")[0].strip())

    for mid, desc in ilsvrc.items():
        d0 = _norm_ic_label(desc.split(",")[0].strip())
        d_full = _norm_ic_label(desc)
        if d0 == n_key or d0 == n_full or d_full == n_full:
            return int(mid)

    best_mid: Optional[int] = None
    best_score = 0
    for mid, desc in ilsvrc.items():
        d0 = _norm_ic_label(desc.split(",")[0].strip())
        if d0.startswith(n_key) or n_key.startswith(d0):
            score = 2
        elif n_key in d0 or d0 in n_key or n_key in _norm_ic_label(desc):
            score = 1
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_mid = int(mid)

    if best_mid is None or best_score == 0:
        raise ValueError(
            f"Could not map dataset class string {class_str!r} to any id2label entry "
            f"(tried {len(ilsvrc)} model classes)."
        )
    return best_mid


def build_image_class_label_remap(
    label_feature: Any,
    id2label: Dict[Any, str],
) -> Optional[List[int]]:
    """Map dataset label ints -> model ``argmax(logits)`` indices.

    ``None`` means identity (``ds_label == model_class_id``). A table is returned when the
    dataset has fewer classes than the head (e.g. Imagenette) or when both have 1000 classes
    but ordering differs (e.g. ``ILSVRC/imagenet-1k`` synset-sorted ids vs torchvision order).
    """
    if label_feature is None or not hasattr(label_feature, "num_classes"):
        return None
    n_ds = int(label_feature.num_classes)
    if not id2label:
        return None
    ilsvrc = {int(k): str(v) for k, v in id2label.items()}
    n_model = len(ilsvrc)
    if n_model == 0 or n_ds == 0:
        return None

    def ds_class_str(i: int) -> str:
        if hasattr(label_feature, "int2str"):
            return str(label_feature.int2str(i))
        names = getattr(label_feature, "names", None)
        if names is not None and i < len(names):
            return str(names[i])
        raise ValueError(f"Cannot resolve dataset class name for index {i}")

    table: List[int] = []
    for i in range(n_ds):
        table.append(_match_class_description_to_model_id(ds_class_str(i), ilsvrc))

    if table == list(range(n_ds)):
        return None
    return table


def image_classification_metric_labels(
    references: Iterable[int],
    label_feature: Any,
    model_config: Any,
) -> List[int]:
    refs_list = [int(r) for r in references]
    id2label = getattr(model_config, "id2label", None)
    if not id2label:
        return refs_list
    table = build_image_class_label_remap(label_feature, id2label)
    if table is None:
        return refs_list
    return [table[r] for r in refs_list]
