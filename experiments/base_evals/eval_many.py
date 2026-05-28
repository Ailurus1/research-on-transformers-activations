
from __future__ import annotations

import gc
import itertools
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from acta import AutoAnalyzer

_EXAMPLES_DIR = Path(__file__).resolve().parent
import sys

if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from audio_samples import load_audio_samples as _load_shared_audio_samples
from image_samples import load_image_samples as _load_shared_image_samples
from target_layers import target_layers_for_model

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "This script requires the `datasets` package. Install with: pip install datasets"
    ) from exc

MODEL_SPECS: List[Tuple[str, str]] = [
    ("masked-language-modeling", "albert/albert-base-v2"),
    ("masked-language-modeling", "microsoft/deberta-v3-base"),
    ("text-generation", "gpt2-medium"),
    ("text-generation", "bigscience/bloom-560m"),
    ("seq2seq", "facebook/bart-base"),
    ("seq2seq", "t5-base"),
    ("image-classification", "google/vit-base-patch16-224"),
    ("image-classification", "microsoft/swin-tiny-patch4-window7-224"),
    ("image-generation", "openai/imagegpt-small"),
    ("image-captioning", "Salesforce/blip-image-captioning-base"),
    ("automatic-speech-recognition", "openai/whisper-base"),
    ("speech-representation", "facebook/hubert-base-ls960"),
    ("speech-representation", "microsoft/unispeech-sat-base"),
]

WIKITEXT_CONFIG = "wikitext-2-raw-v1"

DOMAIN_DATASETS: Dict[str, Dict[str, str]] = {
    "text": {"dataset": "Salesforce/wikitext", "config": WIKITEXT_CONFIG},
    "vision": {"dataset": "ILSVRC/imagenet-1k", "config": "(default)"},
    "audio": {"dataset": "librispeech_asr", "config": "clean"},
}

NUM_SAMPLES = 32
MAX_TEXT_CHARS = 512
MIN_TEXT_CHARS = 40
MAX_NEW_TOKENS = 24
DUMP_ROOT = Path(".acta_dump_results/eval_many")

MPS_CPU_MODEL_IDS = frozenset(
    {
        "gpt2-medium",
        "bigscience/bloom-560m",
    }
)

TEXT_TASKS = frozenset(
    {"masked-language-modeling", "text-generation", "seq2seq"}
)
VISION_TASKS = frozenset(
    {"image-classification", "image-generation", "image-captioning"}
)
AUDIO_TASKS = frozenset(
    {"automatic-speech-recognition", "speech-representation"}
)


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_device(model_id: str, preferred: torch.device) -> torch.device:
    if preferred.type == "mps" and model_id in MPS_CPU_MODEL_IDS:
        return torch.device("cpu")
    return preferred


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


def _slug(model_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))
    return safe.strip("_") or "model"


def _domain_for_task(task: str) -> str:
    if task in TEXT_TASKS:
        return "text"
    if task in VISION_TASKS:
        return "vision"
    if task in AUDIO_TASKS:
        return "audio"
    raise ValueError(f"Unknown task: {task!r}")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_dataset_rows(
    dataset: str,
    config: str,
    split: str,
    n: int,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config == "(default)":
        stream = load_dataset(dataset, **kwargs)
    else:
        stream = load_dataset(dataset, config, **kwargs)
    return list(itertools.islice(stream, n))


def _is_usable_wikitext_line(text: str) -> bool:
    if len(text) < MIN_TEXT_CHARS:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("=") and stripped.endswith("="):
        return False
    return True


def load_text_samples(n: int) -> list[str]:
    spec = DOMAIN_DATASETS["text"]
    kwargs: dict[str, Any] = {"split": "test", "streaming": True}
    stream = load_dataset(spec["dataset"], spec["config"], **kwargs)
    texts: list[str] = []
    for row in stream:
        text = str(row.get("text", "")).strip()[:MAX_TEXT_CHARS]
        if not _is_usable_wikitext_line(text):
            continue
        texts.append(text)
        if len(texts) >= n:
            break
    if len(texts) < n:
        raise RuntimeError(
            f"Only found {len(texts)} usable wikitext lines (need {n}). "
            "Try lowering MIN_TEXT_CHARS."
        )
    return texts


def load_vision_samples(n: int) -> list[Any]:
    return _load_shared_image_samples(n)


def load_audio_samples(n: int) -> list[dict[str, Any]]:
    return _load_shared_audio_samples(n)


def load_hf_bundle(
    task: str, model_id: str, device: torch.device
) -> tuple[nn.Module, Any | None, Any | None]:
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoProcessor,
        AutoTokenizer,
        BlipForConditionalGeneration,
        BlipProcessor,
        ImageGPTForCausalImageModeling,
        ImageGPTImageProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    dtype = torch.float32
    tokenizer: Any | None = None
    processor: Any | None = None

    if task == "masked-language-modeling":
        if "deberta" in model_id.lower():
            from transformers import DebertaV2Tokenizer

            tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=dtype)
    elif task == "text-generation":
        extra: dict[str, Any] = {}
        if "bloom" in model_id.lower():
            extra["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_id, **extra)
        _ensure_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, **extra
        )
    elif task == "seq2seq":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _ensure_pad_token(tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype)
    elif task == "image-classification":
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(
            model_id, dtype=dtype
        )
    elif task == "image-generation":
        processor = ImageGPTImageProcessor.from_pretrained(model_id)
        model = ImageGPTForCausalImageModeling.from_pretrained(
            model_id, dtype=dtype
        )
    elif task == "image-captioning":
        processor = BlipProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype
        )
    elif task == "automatic-speech-recognition":
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype
        )
    elif task == "speech-representation":
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=dtype)
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    return model.to(device).eval(), tokenizer, processor


def _run_text_forward(
    wrapped: nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
) -> None:
    for text in texts:
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        if batch["input_ids"].shape[-1] == 0:
            continue
        with torch.inference_mode():
            wrapped(**_to_device(dict(batch), device))


def _run_text_generate(
    wrapped: nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    prefix: str | None = None,
) -> None:
    for text in texts:
        prompt = f"{prefix}{text}" if prefix else text
        batch = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        if batch["input_ids"].shape[-1] == 0:
            continue
        with torch.inference_mode():
            wrapped.generate(
                **_to_device(dict(batch), device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )


def _run_vision(
    wrapped: nn.Module,
    processor: Any,
    images: list[Any],
    device: torch.device,
    *,
    use_generate: bool = False,
) -> None:
    for image in images:
        batch = processor(images=image, return_tensors="pt")
        batch = _to_device(dict(batch), device)
        with torch.inference_mode():
            if use_generate:
                wrapped.generate(**batch, max_new_tokens=MAX_NEW_TOKENS)
            else:
                wrapped(**batch)


def _run_audio_forward(
    wrapped: nn.Module,
    processor: Any,
    clips: list[dict[str, Any]],
    device: torch.device,
) -> None:
    for clip in clips:
        batch = processor(
            clip["array"],
            sampling_rate=clip["sampling_rate"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            wrapped(**_to_device(dict(batch), device))


def _run_whisper_generate(
    wrapped: nn.Module,
    processor: Any,
    clips: list[dict[str, Any]],
    device: torch.device,
) -> None:
    for clip in clips:
        batch = processor(
            clip["array"],
            sampling_rate=clip["sampling_rate"],
            return_tensors="pt",
        )
        batch = _to_device(dict(batch), device)
        with torch.inference_mode():
            wrapped.generate(
                input_features=batch["input_features"],
                max_new_tokens=MAX_NEW_TOKENS,
            )


def evaluate_model(
    task: str,
    model_id: str,
    device: torch.device,
    *,
    samples: Any,
) -> Path | None:
    domain = _domain_for_task(task)
    slug = _slug(model_id)
    dump_path = DUMP_ROOT / f"{slug}_{task.replace('-', '_')}"
    print(f"\n=== {task} :: {model_id} ({domain}) ===", flush=True)

    run_device = _model_device(model_id, device)
    if run_device != device:
        print(f"Using {run_device} for {model_id} (MPS workaround)", flush=True)
    base, tokenizer, processor = load_hf_bundle(task, model_id, run_device)
    try:
        tl = target_layers_for_model(model_id)
    except ValueError:
        tl = None
    wrapped = AutoAnalyzer(
        base,
        dump_stats_path=str(dump_path),
        target_layers=tl,
        draw_charts=True,
        draw_attention_maps=True,
        verbose=True,
        tokenizer=tokenizer,
        vit_reg_patch_labels=task in VISION_TASKS,
        asr_chunk_labels=task in AUDIO_TASKS,
        finalize_on_exit=False,
    )
    wrapped.eval()

    try:
        if domain == "text":
            texts = samples
            assert tokenizer is not None
            if task == "masked-language-modeling":
                _run_text_forward(wrapped, tokenizer, texts, run_device)
            elif task == "text-generation":
                _run_text_generate(wrapped, tokenizer, texts, run_device)
            else:
                prefix = "summarize: " if "t5" in model_id.lower() else None
                _run_text_generate(
                    wrapped, tokenizer, texts, run_device, prefix=prefix
                )
        elif domain == "vision":
            images = samples
            assert processor is not None
            use_generate = task in ("image-captioning",)
            _run_vision(
                wrapped, processor, images, run_device, use_generate=use_generate
            )
        else:
            clips = samples
            assert processor is not None
            if task == "automatic-speech-recognition":
                _run_whisper_generate(wrapped, processor, clips, run_device)
            else:
                _run_audio_forward(wrapped, processor, clips, run_device)
    finally:
        wrapped._finalize_on_exit()
        wrapped.unregister_hooks()

    out_dir = Path(wrapped.output_run_dir)
    print(f"Wrote results to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    _quiet_hf()
    device = _preferred_device()
    print(f"Device: {device}", flush=True)
    print(f"Samples per dataset: {NUM_SAMPLES}", flush=True)

    cache: dict[str, Any] = {}
    failures: list[tuple[str, str, str]] = []

    for task, model_id in MODEL_SPECS:
        domain = _domain_for_task(task)
        try:
            if domain not in cache:
                print(f"Loading {domain} dataset …", flush=True)
                if domain == "text":
                    cache[domain] = load_text_samples(NUM_SAMPLES)
                elif domain == "vision":
                    cache[domain] = load_vision_samples(NUM_SAMPLES)
                else:
                    cache[domain] = load_audio_samples(NUM_SAMPLES)
            evaluate_model(task, model_id, device, samples=cache[domain])
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {model_id}: {msg}", flush=True)
            failures.append((task, model_id, msg))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("\n=== Summary ===", flush=True)
    ok = len(MODEL_SPECS) - len(failures)
    print(f"Completed: {ok}/{len(MODEL_SPECS)}", flush=True)
    for task, model_id, msg in failures:
        print(f"  - [{task}] {model_id}: {msg}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
