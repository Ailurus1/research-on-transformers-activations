from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import evaluate
import torch
from datasets import load_dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForSpeechSeq2Seq,
    DataCollatorForLanguageModeling,
    AutoProcessor,
    AutoTokenizer,
)

from experiments.aggregate_size_effect_results import aggregate_size_effect_results
from experiments.cache_cleanup import (
    clear_hf_dataset_cache,
    release_memory,
    remove_hf_hub_model_cache,
)
from experiments.utils import image_classification_metric_labels, set_seed

logger = logging.getLogger(__name__)


SIZE_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "masked-language-modeling": {
        "albert": [
            "albert/albert-base-v2",
            "albert/albert-large-v2",
            "albert/albert-xlarge-v2",
            "albert/albert-xxlarge-v2",
        ]
    },
    "text-generation": {
        "gpt2": [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        ]
    },
    "image-classification": {
        "deit": [
            "facebook/deit-tiny-patch16-224",
            "facebook/deit-small-patch16-224",
            "facebook/deit-base-patch16-224",
            "facebook/deit-large-patch16-224",
        ]
    },
    "automatic-speech-recognition": {
        "whisper": [
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-large",
        ]
    },
}


TASKS: Dict[str, Dict[str, Any]] = {
    "masked-language-modeling": {
        "dataset": ("wikitext", "wikitext-103-raw-v1"),
        "split": "validation",
        "text_col": "text",
        "metric": "mlm_loss",
        "max_length": 256,
        "mlm_probability": 0.15,
    },
    "text-generation": {
        "dataset": ("wikitext", "wikitext-103-raw-v1"),
        "split": "validation",
        "text_col": "text",
        "metric": "perplexity",
    },
    "image-classification": {
        "dataset": ("ILSVRC/imagenet-1k", None),
        "split": "validation",
        "image_col": "image",
        "label_col": "label",
        "metric": "accuracy",
    },
    "automatic-speech-recognition": {
        "dataset": ("librispeech_asr", "clean"),
        "split": "validation",
        "audio_col": "audio",
        "text_col": "text",
        "metric": "wer",
    },
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_load_kwargs() -> Dict[str, Any]:
    return {"low_cpu_mem_usage": True}


def _load_dataset_head(
    path: str, config: Optional[str], split_name: str, max_samples: Optional[int]
):
    split_arg = (
        split_name if max_samples is None else f"{split_name}[:{max(1, max_samples)}]"
    )
    return load_dataset(path, config, split=split_arg, trust_remote_code=True)


def _iter_chunked(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _wrap_mlm_core_ids(tokenizer: Any, core: List[int]) -> List[int]:
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is not None and sep_id is not None:
        return [cls_id] + list(core) + [sep_id]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is not None and eos_id is not None:
        return [bos_id] + list(core) + [eos_id]
    if bos_id is not None:
        return [bos_id] + list(core)
    return list(core)


def _mlm_token_windows_from_texts(
    tokenizer: Any,
    texts: Iterable[str],
    max_length: int,
    min_tail_tokens: int = 32,
) -> List[Dict[str, List[int]]]:
    """Merge WikiText lines into fixed-length windows so MLM masking almost always has targets.

    PyTorch cross-entropy returns NaN when every label is ``ignore_index`` (-100). With
    ``batch_size=1`` and many very short rows, ``DataCollatorForLanguageModeling`` can mask
    zero tokens, leaving no supervised positions.
    """
    n_special = len(_wrap_mlm_core_ids(tokenizer, []))
    core_budget = max(1, int(max_length) - n_special)
    buffer: List[int] = []
    windows: List[Dict[str, List[int]]] = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = enc.get("input_ids") or []
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= core_budget:
            core = buffer[:core_budget]
            buffer = buffer[core_budget:]
            packed = _wrap_mlm_core_ids(tokenizer, core)
            if len(packed) > max_length:
                packed = packed[:max_length]
            windows.append({"input_ids": packed, "attention_mask": [1] * len(packed)})

    if len(buffer) >= min_tail_tokens:
        packed = _wrap_mlm_core_ids(tokenizer, buffer)
        buffer.clear()
        if len(packed) > max_length:
            packed = packed[:max_length]
        windows.append({"input_ids": packed, "attention_mask": [1] * len(packed)})

    return windows


def _quantize_linear_asymmetric_int8(model: nn.Module) -> nn.Module:
    qmodel = copy.deepcopy(model)
    _replace_linear_with_fake_quant(qmodel)
    return qmodel


def _fake_quantize_affine(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    if not x.is_floating_point():
        return x
    qmin = 0
    qmax = (1 << num_bits) - 1
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    if torch.isclose(x_max, x_min):
        return x
    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return (q - zero_point) * scale


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.register_buffer(
            "q_weight",
            _fake_quantize_affine(linear.weight.detach().to(torch.float32)),
        )
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = _fake_quantize_affine(x)
        out = torch.nn.functional.linear(x_q, self.q_weight, self.bias)
        return _fake_quantize_affine(out)


def _replace_linear_with_fake_quant(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child))
        else:
            _replace_linear_with_fake_quant(child)


def eval_masked_language_modeling(
    model: nn.Module,
    tokenizer: Any,
    max_samples: Optional[int],
    batch_size: int,
    compute_device: torch.device,
) -> Dict[str, Any]:
    task = TASKS["masked-language-modeling"]
    ds = _load_dataset_head(task["dataset"][0], task["dataset"][1], task["split"], max_samples)
    texts = [t for t in ds[task["text_col"]] if isinstance(t, str) and t.strip()]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(task["mlm_probability"]),
    )
    max_length = int(task["max_length"])
    windows = _mlm_token_windows_from_texts(tokenizer, texts, max_length)
    if not windows:
        raise RuntimeError(
            "No MLM windows built from WikiText (empty split, wrong column, or tokenizer issue)."
        )

    losses: List[float] = []

    model.eval()
    with torch.no_grad():
        for window_batch in tqdm(
            _iter_chunked(windows, batch_size),
            total=math.ceil(len(windows) / batch_size),
            desc="masked-language-modeling",
            leave=False,
        ):
            features = []
            for w in window_batch:
                features.append(
                    {
                        "input_ids": torch.tensor(w["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(w["attention_mask"], dtype=torch.long),
                    }
                )
            batch = collator(features)
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            out = model(**batch)
            loss_v = float(out.loss.item())
            if math.isfinite(loss_v):
                losses.append(loss_v)
            else:
                logger.warning("Skipping non-finite MLM loss for one batch (no masked labels).")
            del batch, out

    if not losses:
        raise RuntimeError("All MLM batches produced non-finite loss; try lowering mlm_probability.")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"mlm_loss": avg_loss, "perplexity": perplexity}


def eval_text_generation(
    model: nn.Module,
    tokenizer: Any,
    max_samples: Optional[int],
    batch_size: int,
    compute_device: torch.device,
) -> Dict[str, Any]:
    task = TASKS["text-generation"]
    texts: List[str] = []
    if max_samples is None:
        ds = load_dataset(
            task["dataset"][0],
            task["dataset"][1],
            split=task["split"],
            trust_remote_code=True,
        )
        texts = [row[task["text_col"]] for row in ds if row.get(task["text_col"])]
    else:
        ds_stream = load_dataset(
            task["dataset"][0],
            task["dataset"][1],
            split=task["split"],
            streaming=True,
            trust_remote_code=True,
        )
        for row in ds_stream:
            t = row.get(task["text_col"])
            if t:
                texts.append(t)
            if len(texts) >= max_samples:
                break

    losses: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch_texts in tqdm(
            _iter_chunked(texts, batch_size),
            total=math.ceil(len(texts) / batch_size),
            desc="text-generation",
            leave=False,
        ):
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(compute_device)
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss.item()))
            del enc, out

    avg_nll = sum(losses) / max(len(losses), 1)
    perplexity = math.exp(avg_nll) if avg_nll < 20 else float("inf")
    return {"perplexity": perplexity, "avg_nll": avg_nll}


def eval_image_classification(
    model: nn.Module,
    processor: Any,
    max_samples: Optional[int],
    batch_size: int,
    compute_device: torch.device,
) -> Dict[str, Any]:
    task = TASKS["image-classification"]
    ds = _load_dataset_head(task["dataset"][0], task["dataset"][1], task["split"], max_samples)
    metric = evaluate.load(task["metric"])

    images = ds[task["image_col"]]
    refs = ds[task["label_col"]]
    label_feature = ds.features.get(task["label_col"])
    refs_for_metric = image_classification_metric_labels(refs, label_feature, model.config)
    preds: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_chunked(images, batch_size),
            total=math.ceil(len(images) / batch_size),
            desc="image-classification",
            leave=False,
        ):
            inp = processor(images=batch, return_tensors="pt").to(compute_device)
            out = model(**inp).logits
            preds.extend(torch.argmax(out, dim=-1).cpu().tolist())
            del inp, out

    return metric.compute(predictions=preds, references=refs_for_metric)


def eval_asr(
    model: nn.Module,
    processor: Any,
    max_samples: Optional[int],
    batch_size: int,
    compute_device: torch.device,
) -> Dict[str, Any]:
    task = TASKS["automatic-speech-recognition"]
    ds = _load_dataset_head(task["dataset"][0], task["dataset"][1], task["split"], max_samples)
    metric = evaluate.load(task["metric"])

    audios = ds[task["audio_col"]]
    refs = ds[task["text_col"]]
    preds: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_chunked(audios, batch_size),
            total=math.ceil(len(audios) / batch_size),
            desc="automatic-speech-recognition",
            leave=False,
        ):
            for item in batch:
                inp = processor(
                    item["array"],
                    sampling_rate=item["sampling_rate"],
                    return_tensors="pt",
                ).to(compute_device)
                if hasattr(model, "generate"):
                    gen_kw: Dict[str, Any] = {"max_new_tokens": 96}
                    if getattr(getattr(model, "config", None), "model_type", None) == "whisper":
                        gen_kw["language"] = "en"
                        gen_kw["task"] = "transcribe"
                    gen = model.generate(**inp, **gen_kw)
                    preds.append(processor.batch_decode(gen, skip_special_tokens=True)[0])
                    del gen
                else:
                    logits = model(**inp).logits
                    pred_ids = torch.argmax(logits, dim=-1)
                    preds.extend(processor.batch_decode(pred_ids))
                    del logits, pred_ids
                del inp

    return metric.compute(predictions=preds, references=refs)


def _build_model_and_processor(domain: str, model_id: str, quantized: bool):
    base_device = _device()

    if domain == "masked-language-modeling":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id, **_model_load_kwargs())
        processor = tokenizer
    elif domain == "text-generation":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, **_model_load_kwargs())
        processor = tokenizer
    elif domain == "image-classification":
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, **_model_load_kwargs())
    elif domain == "automatic-speech-recognition":
        processor = AutoProcessor.from_pretrained(model_id)
        if "whisper" in model_id.lower():
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **_model_load_kwargs())
        else:
            model = AutoModelForCTC.from_pretrained(model_id, **_model_load_kwargs())
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    if quantized:
        model = _quantize_linear_asymmetric_int8(model)
        model = model.to(base_device)
        compute_device = base_device
    else:
        model = model.to(base_device)
        compute_device = base_device

    return model, processor, compute_device


def _evaluate_once(
    domain: str,
    model_id: str,
    quantized: bool,
    max_samples: Optional[int],
    batch_size: int,
    run_dir: Path,
) -> Dict[str, Any]:
    suffix = "int8_fake_static" if quantized else "fp32"
    (run_dir / suffix).mkdir(parents=True, exist_ok=True)

    model, processor, compute_device = _build_model_and_processor(domain, model_id, quantized)

    if domain == "masked-language-modeling":
        score = eval_masked_language_modeling(model, processor, max_samples, batch_size, compute_device)
        metric_name = TASKS[domain]["metric"]
    elif domain == "text-generation":
        score = eval_text_generation(model, processor, max_samples, batch_size, compute_device)
        metric_name = TASKS[domain]["metric"]
    elif domain == "image-classification":
        score = eval_image_classification(model, processor, max_samples, batch_size, compute_device)
        metric_name = TASKS[domain]["metric"]
    elif domain == "automatic-speech-recognition":
        score = eval_asr(model, processor, max_samples, batch_size, compute_device)
        metric_name = TASKS[domain]["metric"]
    else:
        raise ValueError(domain)

    return {
        "status": "ok",
        "precision": suffix,
        "metric": metric_name,
        "score": score,
        "device": str(compute_device),
    }


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    if args.batch_size != 1:
        logger.warning("Forcing batch_size=1 (CLI had --batch-size=%d)", args.batch_size)

    selected_domains = list(SIZE_GROUPS.keys()) if "all" in args.tasks else args.tasks
    report: Dict[str, Any] = {
        "seed": args.seed,
        "max_samples": args.max_samples if args.max_samples is not None else "full",
        "batch_size": batch_size,
        "tasks": selected_domains,
        "results": {},
    }

    for domain in selected_domains:
        report["results"][domain] = {}
        for family, model_ids in SIZE_GROUPS[domain].items():
            logger.info("Domain=%s Family=%s", domain, family)
            report["results"][domain][family] = {}
            for model_id in model_ids:
                model_safe = model_id.replace("/", "__")
                run_dir = output_root / domain / family / model_safe
                run_dir.mkdir(parents=True, exist_ok=True)
                report["results"][domain][family][model_id] = {}

                for quantized in (False, True):
                    precision = "int8_fake_static" if quantized else "fp32"
                    logger.info("Evaluating %s (%s)", model_id, precision)
                    try:
                        result = _evaluate_once(
                            domain=domain,
                            model_id=model_id,
                            quantized=quantized,
                            max_samples=args.max_samples,
                            batch_size=batch_size,
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        result = {"status": "failed", "precision": precision, "error": str(exc)}
                        logger.exception("Failed: %s (%s)", model_id, precision)
                    finally:
                        release_memory()

                    report["results"][domain][family][model_id][precision] = result
                    with open(run_dir / f"{precision}_result.json", "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)

                remove_hf_hub_model_cache(model_id)
                release_memory()

        ds_spec = TASKS[domain]["dataset"]
        clear_hf_dataset_cache(ds_spec[0], ds_spec[1])
        release_memory()

        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if not args.skip_aggregation:
            try:
                aggregate_size_effect_results(output_root.resolve())
                logger.info("Plots and aggregated_results.json updated after domain=%s", domain)
            except Exception:
                logger.exception(
                    "aggregate_size_effect_results failed after domain=%s (summary.json is still valid)",
                    domain,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore effect of model size + int8 quantization.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap examples per task split; omit for the full split (can be large / memory-heavy).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Ignored: batch size is always 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/size_effect")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Do not build plots or aggregated_results.json after each domain.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", *SIZE_GROUPS.keys()],
        help="Task domains to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
