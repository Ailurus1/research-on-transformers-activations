from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from target_layers import GPT2
from text_prompts import prompts


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = _preferred_device()
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoAnalyzer(
        model,
        tokenizer=tokenizer,
        target_layers=GPT2,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
    )

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
            model.generate(
                **{k: v.to(device) for k, v in batch.items()},
                max_new_tokens=24,
                do_sample=False,
            )


if __name__ == "__main__":
    main()
