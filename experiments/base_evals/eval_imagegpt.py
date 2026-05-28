from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import ImageGPTForCausalImageModeling, ImageGPTImageProcessor

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from image_samples import load_image_samples
from target_layers import IMAGEGPT_SMALL


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _quiet_hf() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as tr_logging

        tr_logging.set_verbosity_error()
    except Exception:
        pass


def main() -> None:
    _quiet_hf()
    model_name = "openai/imagegpt-small"
    device = _preferred_device()

    processor = ImageGPTImageProcessor.from_pretrained(model_name)
    base_model = ImageGPTForCausalImageModeling.from_pretrained(model_name).to(device)
    model = AutoAnalyzer(
        base_model,
        target_layers=IMAGEGPT_SMALL,
        draw_charts=True,
        draw_attention_maps=False,
        verbose=True,
        vit_reg_patch_labels=True,
        finalize_on_exit=True,
    )
    model.eval()

    for image in load_image_samples(5):
        batch = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            model(**{k: v.to(device) for k, v in batch.items()})


if __name__ == "__main__":
    main()
