from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from target_layers import T5_BASE
from text_prompts import prompts

T5_PREFIX = "summarize: "


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    model_name = "t5-base"
    device = _preferred_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    model = AutoAnalyzer(
        base_model,
        tokenizer=tokenizer,
        target_layers=T5_BASE,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
    )
    model.eval()

    for text in prompts:
        batch = tokenizer(
            f"{T5_PREFIX}{text}",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        if batch["input_ids"].shape[-1] == 0:
            continue
        with torch.inference_mode():
            model.generate(
                **{k: v.to(device) for k, v in batch.items()},
                max_new_tokens=24,
                do_sample=False,
            )


if __name__ == "__main__":
    main()
