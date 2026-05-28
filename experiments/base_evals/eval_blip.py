from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from image_samples import load_image_samples
from target_layers import BLIP_CAPTIONING

MAX_NEW_TOKENS = 24


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
    model_name = "Salesforce/blip-image-captioning-base"
    device = _preferred_device()

    processor = BlipProcessor.from_pretrained(model_name)
    base_model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model = AutoAnalyzer(
        base_model,
        tokenizer=processor.tokenizer,
        target_layers=BLIP_CAPTIONING,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
        vit_reg_patch_labels=True,
    )
    model.eval()

    for image in load_image_samples():
        batch = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            model.generate(
                **{k: v.to(device) for k, v in batch.items()},
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )


if __name__ == "__main__":
    main()
