from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from image_samples import load_image_samples
from target_layers import VIT_DEIT_LAYERS


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
    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def main() -> None:
    _quiet_hf()

    model_name = "facebook/deit-tiny-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    base_model = AutoModelForImageClassification.from_pretrained(model_name)

    model = AutoAnalyzer(
        base_model,
        dump_stats_path="./vit_activations_analysis",
        target_layers=VIT_DEIT_LAYERS,
        draw_charts=True,
        verbose=True,
        tokenizer=None,
        vit_reg_patch_labels=True,
        asr_chunk_labels=False,
    )
    model.eval()
    device = _preferred_device()
    model.to(device)

    for image in load_image_samples():
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            _ = model(**{k: v.to(device) for k, v in inputs.items()})


if __name__ == "__main__":
    main()
