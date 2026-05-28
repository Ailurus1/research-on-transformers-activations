from __future__ import annotations

import warnings
from typing import Any

from PIL import Image

NUM_IMAGE_SAMPLES = 32

IMAGENET_DATASET = "ILSVRC/imagenet-1k"
CIFAR10_DATASET = "uoft-cs/cifar10"


def _pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected PIL.Image, got {type(image)!r}")


def _load_imagenet_stream(n: int) -> list[Image.Image]:
    from datasets import load_dataset

    stream = load_dataset(IMAGENET_DATASET, split="validation", streaming=True)
    images: list[Image.Image] = []
    for row in stream:
        raw = row.get("image")
        if raw is None:
            continue
        images.append(_pil_rgb(raw))
        if len(images) >= n:
            break
    if len(images) < n:
        raise RuntimeError(f"ImageNet: only loaded {len(images)} images (need {n})")
    return images


def _load_cifar10_hf(n: int) -> list[Image.Image]:
    from datasets import load_dataset

    stream = load_dataset(CIFAR10_DATASET, split="test", streaming=True)
    images: list[Image.Image] = []
    for row in stream:
        raw = row.get("img")
        if raw is None:
            continue
        images.append(_pil_rgb(raw))
        if len(images) >= n:
            break
    if len(images) < n:
        raise RuntimeError(f"CIFAR-10 (HF): only loaded {len(images)} images (need {n})")
    return images


def _load_cifar10_torchvision(n: int, *, root: str | None = None) -> list[Image.Image]:
    from torchvision.datasets import CIFAR10

    cache_root = root or ".acta_cache/cifar10"
    dataset = CIFAR10(root=cache_root, train=False, download=True)
    return [_pil_rgb(dataset[i][0]) for i in range(n)]


def load_image_samples(n: int = NUM_IMAGE_SAMPLES) -> list[Image.Image]:
    """Load ``n`` RGB images, preferring ImageNet-1k validation then CIFAR-10."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    errors: list[str] = []
    loaders: list[tuple[str, Any]] = [
        (f"ImageNet ({IMAGENET_DATASET})", _load_imagenet_stream),
        (f"HF CIFAR-10 ({CIFAR10_DATASET})", _load_cifar10_hf),
        ("torchvision CIFAR-10", _load_cifar10_torchvision),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, loader in loaders:
            try:
                return loader(n)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
    detail = "; ".join(errors)
    raise RuntimeError(f"Could not load {n} images. {detail}")


# assert NUM_IMAGE_SAMPLES == 32
