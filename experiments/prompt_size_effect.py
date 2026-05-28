from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator
from tqdm import tqdm

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from target_layers import BLOOM_560M, GPT2

PROMPT_SIZES = (256, 512, 1024, 2048)
NUM_SEQUENCES = 16
MIN_SEED_CHARS = 80
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"

OPT_125M: list[str] = [
    "model.decoder.layers.*.self_attn.q_proj",
    "model.decoder.layers.*.self_attn.k_proj",
    "model.decoder.layers.*.self_attn.v_proj",
    "model.decoder.layers.*.self_attn.out_proj",
    "model.decoder.layers.*.fc1",
    "model.decoder.layers.*.fc2",
]

MODEL_SPECS: list[tuple[str, list[str]]] = [
    ("openai-community/gpt2", GPT2),
    ("facebook/opt-125m", OPT_125M),
    ("bigscience/bloom-560m", BLOOM_560M),
]

DEFAULT_OUTPUT_JSON = Path("prompt_size_effect_results.json")

KNOWN_MAX_POSITIONS: dict[str, int] = {
    "bigscience/bloom-560m": 2048,
    "bigscience/bloom-1b1": 2048,
    "bigscience/bloom-1b7": 2048,
    "bigscience/bloom-3b": 2048,
    "bigscience/bloom-7b1": 2048,
}


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


def _model_max_positions(model_id: str) -> int:
    mid = model_id.strip()
    mid_low = mid.lower()
    if mid in KNOWN_MAX_POSITIONS:
        return KNOWN_MAX_POSITIONS[mid]
    if mid_low in KNOWN_MAX_POSITIONS:
        return KNOWN_MAX_POSITIONS[mid_low]
    for key, limit in KNOWN_MAX_POSITIONS.items():
        if mid_low.endswith(key.split("/")[-1]) or key in mid_low:
            return limit

    extra: dict[str, Any] = {}
    if "bloom" in mid_low:
        extra["trust_remote_code"] = True
    config = AutoConfig.from_pretrained(model_id, **extra)
    for attr in (
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_seq_len",
        "n_ctx",
    ):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    if "bloom" in mid_low:
        return 2048
    return 0


def _iter_wikitext_lines() -> Iterator[str]:
    from datasets import load_dataset

    stream = load_dataset(
        WIKITEXT_DATASET, WIKITEXT_CONFIG, split="test", streaming=True
    )
    while True:
        for row in stream:
            text = str(row.get("text", "")).strip()
            if len(text) < MIN_SEED_CHARS:
                continue
            if text.startswith("=") and text.endswith("="):
                continue
            yield text


def _load_seed_texts(n: int) -> list[str]:
    seeds: list[str] = []
    for line in _iter_wikitext_lines():
        seeds.append(line)
        if len(seeds) >= n:
            break
    if len(seeds) < n:
        raise RuntimeError(f"Only found {len(seeds)} wikitext seeds (need {n})")
    return seeds


def _extend_to_length(
    token_ids: list[int],
    target_len: int,
    tokenizer: Any,
    text_iter: Iterator[str],
) -> list[int]:
    ids = list(token_ids)
    while len(ids) < target_len:
        ids.extend(
            tokenizer.encode(next(text_iter), add_special_tokens=False)
        )
    return ids[:target_len]


def build_nested_token_prompts(
    tokenizer: Any,
    seed_texts: list[str],
    sizes: tuple[int, ...] = PROMPT_SIZES,
) -> list[dict[int, list[int]]]:
    if not sizes:
        raise ValueError("sizes must be non-empty")
    max_size = max(sizes)
    ordered = tuple(sorted(sizes))
    text_iter = _iter_wikitext_lines()
    nested: list[dict[int, list[int]]] = []

    for seed in seed_texts:
        base = tokenizer.encode(seed, add_special_tokens=False)
        full = _extend_to_length(base, max_size, tokenizer, text_iter)
        row = {size: full[:size] for size in ordered}
        for i in range(1, len(ordered)):
            prev_size, curr_size = ordered[i - 1], ordered[i]
            if row[curr_size][:prev_size] != row[prev_size]:
                raise RuntimeError(
                    f"Prefix mismatch for seed {seed[:40]!r}: "
                    f"size {prev_size} is not a prefix of size {curr_size}"
                )
        nested.append(row)
    return nested


def _load_causal_lm(model_id: str, device: torch.device) -> tuple[Any, Any]:
    extra: dict[str, Any] = {}
    if "bloom" in model_id.lower():
        extra["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **extra)
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, **extra
    )
    return model.to(device).eval(), tokenizer


def _sequence_top1(analyzer: Any) -> float:
    if not analyzer._sequences:
        raise RuntimeError("No sequence recorded after forward pass")
    rec = analyzer._sequences[-1]
    report = rec.get("report")
    if not isinstance(report, dict):
        raise RuntimeError("Sequence record has no report")
    abs_tops = report.get("activation_abs")
    if not isinstance(abs_tops, dict):
        raise RuntimeError("Report has no activation_abs tops")
    value = abs_tops.get("top_1")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise RuntimeError("top_1 missing from activation_abs report")
    return float(value)


def collect_top1_for_size(
    wrapped: Any,
    token_prompts: list[list[int]],
    device: torch.device,
) -> list[float]:
    values: list[float] = []
    for token_ids in tqdm(token_prompts):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            wrapped(input_ids=input_ids, attention_mask=attention_mask)
        values.append(_sequence_top1(wrapped))
    return values


def run_experiment(*, output_path: Path = DEFAULT_OUTPUT_JSON) -> dict[str, Any]:
    _quiet_hf()
    device = _preferred_device()
    print(f"Device: {device}")
    print(f"Prompt sizes: {list(PROMPT_SIZES)}")
    print(f"Sequences per size: {NUM_SEQUENCES}\n")

    seed_texts = _load_seed_texts(NUM_SEQUENCES)
    results: dict[str, Any] = {
        "prompt_sizes": list(PROMPT_SIZES),
        "num_sequences": NUM_SEQUENCES,
        "device": str(device),
        "models": [],
    }

    for model_id, target_layers in MODEL_SPECS:
        max_pos = _model_max_positions(model_id)
        sizes_for_model = tuple(s for s in PROMPT_SIZES if s <= max_pos)
        if not sizes_for_model:
            print(
                f"SKIP {model_id}: max context {max_pos}; "
                f"none of {list(PROMPT_SIZES)} fit",
                flush=True,
            )
            results["models"].append(
                {
                    "model_id": model_id,
                    "max_position_embeddings": max_pos,
                    "error": "no applicable prompt sizes",
                }
            )
            continue

        print(f"{model_id} (max {max_pos})")
        model_result: dict[str, Any] = {
            "model_id": model_id,
            "max_position_embeddings": max_pos,
            "prompt_sizes": list(sizes_for_model),
            "by_prompt_size": {},
        }
        try:
            base_model, tokenizer = _load_causal_lm(model_id, device)
            nested = build_nested_token_prompts(
                tokenizer, seed_texts, sizes=sizes_for_model
            )
        except Exception as exc:
            print(f"FAILED to load {model_id}: {exc}", flush=True)
            model_result["error"] = str(exc)
            results["models"].append(model_result)
            continue

        for size in sizes_for_model:
            prompts = [row[size] for row in nested]
            dump_dir = (
                Path(".acta_prompt_size_effect")
                / model_id.replace("/", "__")
                / str(size)
            )
            wrapped = AutoAnalyzer(
                base_model,
                dump_stats_path=str(dump_dir),
                target_layers=target_layers,
                # draw_charts=True,
                # draw_attention_maps=True,
                verbose=False,
                tokenizer=tokenizer,
                finalize_on_exit=False,
            )
            wrapped.eval()
            try:
                top1_values = collect_top1_for_size(wrapped, prompts, device)
            finally:
                wrapped._finalize_on_exit()
                wrapped.unregister_hooks()

            mean = statistics.mean(top1_values)
            std = (
                statistics.stdev(top1_values)
                if len(top1_values) > 1
                else 0.0
            )
            print(
                f"  tokens={size:4d}: "
                f"mean Top-1 |activation| = {mean:.6g}  "
                f"std = {std:.6g}  (n={len(top1_values)})"
            )
            model_result["by_prompt_size"][str(size)] = {
                "top1_per_sequence": top1_values,
                "mean_top1_abs_activation": mean,
                "std_top1_abs_activation": std,
                "n": len(top1_values),
                "dump_dir": str(wrapped.output_run_dir),
            }

        results["models"].append(model_result)
        try:
            del base_model, tokenizer
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print()
        _write_results(output_path, results)

    _write_results(output_path, results)
    print("Done.")
    return results


def _write_results(output_path: Path, results: dict[str, Any]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Top-1 |activation| vs prompt length on small causal LMs."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    args = parser.parse_args()
    run_experiment(output_path=args.output)


if __name__ == "__main__":
    main()
