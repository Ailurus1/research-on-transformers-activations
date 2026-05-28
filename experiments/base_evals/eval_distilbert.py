from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from target_layers import target_layers_for_model
from text_prompts import prompts


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _preferred_device()
    base_model = AutoModel.from_pretrained(model_name).to(device)

    model = AutoAnalyzer(
        base_model,
        dump_stats_path="./distilbert_activations_analysis",
        target_layers=target_layers_for_model(model_name),
        draw_charts=True,
        verbose=True,
        tokenizer=tokenizer,
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
