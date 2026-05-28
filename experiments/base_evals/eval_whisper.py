from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from audio_samples import load_audio_samples
from target_layers import WHISPER_ENCODER

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

    model_name = "openai/whisper-base"
    device = _preferred_device()

    processor = WhisperProcessor.from_pretrained(model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    model = AutoAnalyzer(
        base_model,
        target_layers=WHISPER_ENCODER,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
        tokenizer=None,
        asr_chunk_labels=True,
        vit_reg_patch_labels=False,
        chart_3d_max_tokens=24,
    )
    model.eval()

    for clip in load_audio_samples(8):
        batch = processor(
            clip["array"],
            sampling_rate=clip["sampling_rate"],
            return_tensors="pt",
        )
        input_features = batch.input_features.to(device)
        with torch.inference_mode():
            model.generate(input_features=input_features, max_new_tokens=MAX_NEW_TOKENS)


if __name__ == "__main__":
    main()
