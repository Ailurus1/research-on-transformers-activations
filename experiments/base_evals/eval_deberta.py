from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, DebertaV2Tokenizer

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from target_layers import DEBERTA_V3
from text_prompts import prompts


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_deberta_tokenizer(model_name: str) -> DebertaV2Tokenizer:
    try:
        return DebertaV2Tokenizer.from_pretrained(model_name)
    except ImportError as exc:
        raise SystemExit(
            "DeBERTa tokenizer requires protobuf. Install deps with: "
            "pip install -e .  or  uv pip install protobuf sentencepiece"
        ) from exc


def main() -> None:
    model_name = "microsoft/deberta-v3-base"
    device = _preferred_device()
    tokenizer = _load_deberta_tokenizer(model_name)
    base_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    model = AutoAnalyzer(
        base_model,
        tokenizer=tokenizer,
        target_layers=DEBERTA_V3,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
    )
    model.eval()

    for text in prompts:
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        if batch["input_ids"].shape[-1] == 0:
            continue
        with torch.inference_mode():
            model(**{k: v.to(device) for k, v in batch.items()})


if __name__ == "__main__":
    main()
