from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List

import evaluate
import torch
from acta import AutoAnalyzer
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from experiments.utils import SampleConfig, chunked, sample_dataset, set_seed

logger = logging.getLogger(__name__)


MODEL_GROUPS: Dict[str, List[str]] = {
    "text-classification": [
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "albert/albert-base-v2",
        "google-bert/bert-base-uncased",
        "answerdotai/ModernBERT-base",
    ],
    "machine-translation": [
        "google-t5/t5-small",
        "google/mt5-small",
        "facebook/bart-base",
    ],
    "text-generation": [
        "gpt2",
        "google/gemma-2-2b-it",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    "image-classification": [
        "google/vit-base-patch16-224",
        "google/vit-base-patch32-224",
        "facebook/deit-tiny-patch16-224",
    ],
    "automatic-speech-recognition": [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h",
    ],
}


TASKS = {
    "text-classification": {
        "dataset": ("imdb", None),
        "text_col": "text",
        "label_col": "label",
        "metric": "accuracy",
        "sample_split": "test",
    },
    "machine-translation": {
        "dataset": ("Muennighoff/flores200", "eng_Latn-deu_Latn"),
        "src_col": "sentence_eng_Latn",
        "tgt_col": "sentence_deu_Latn",
        "metric": "sacrebleu",
        "sample_split": "dev",
    },
    "text-generation": {
        "dataset": ("allenai/c4", "en"),
        "text_col": "text",
        "metric": "perplexity",
        "sample_split": "validation",
    },
    "image-classification": {
        "dataset": ("imagenet-1k", None),
        "image_col": "image",
        "label_col": "label",
        "metric": "accuracy",
        "sample_split": "validation",
    },
    "automatic-speech-recognition": {
        "dataset": ("librispeech_asr", "clean"),
        "audio_col": "audio",
        "text_col": "text",
        "metric": "wer",
        "sample_split": "validation",
    },
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_text_classification(
    model_id: str, max_samples: int, batch_size: int, out_dir: Path
) -> Dict:
    task = TASKS["text-classification"]
    logger.info("Loading dataset for text classification: %s", task["dataset"][0])
    ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["sample_split"])
    ds = sample_dataset(ds, SampleConfig(split=task["sample_split"], max_samples=max_samples))

    logger.info("Loading model/tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(model, dump_stats_path=str(out_dir / "acta"), tokenizer=tokenizer)
    metric = evaluate.load(task["metric"])

    texts = ds[task["text_col"]]
    labels = ds[task["label_col"]]
    all_preds: List[int] = []
    model.eval()
    batches = list(chunked(texts, batch_size=batch_size))
    with torch.no_grad():
        for batch_texts in tqdm(
            batches,
            desc=f"[text-classification] {model_id}",
            leave=False,
        ):
            enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(_device())
            logits = model(**enc).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    score = metric.compute(predictions=all_preds, references=labels)
    return {"metric": task["metric"], "score": score}


def evaluate_machine_translation(
    model_id: str, max_samples: int, batch_size: int, out_dir: Path
) -> Dict:
    task = TASKS["machine-translation"]
    logger.info("Loading dataset for translation: %s", task["dataset"][0])
    ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["sample_split"])
    ds = sample_dataset(ds, SampleConfig(split=task["sample_split"], max_samples=max_samples))

    logger.info("Loading model/tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(model, dump_stats_path=str(out_dir / "acta"), tokenizer=tokenizer)
    metric = evaluate.load(task["metric"])

    src_texts = ds[task["src_col"]]
    ref_texts = [[x] for x in ds[task["tgt_col"]]]
    preds: List[str] = []
    model.eval()
    batches = list(chunked(src_texts, batch_size=batch_size))
    with torch.no_grad():
        for batch_texts in tqdm(
            batches,
            desc=f"[machine-translation] {model_id}",
            leave=False,
        ):
            enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(_device())
            generated = model.generate(**enc, max_new_tokens=80)
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    score = metric.compute(predictions=preds, references=ref_texts)
    return {"metric": task["metric"], "score": score}


def evaluate_text_generation(
    model_id: str, max_samples: int, batch_size: int, out_dir: Path
) -> Dict:
    task = TASKS["text-generation"]
    logger.info("Loading dataset for generation: %s", task["dataset"][0])
    ds_stream = load_dataset(
        task["dataset"][0],
        task["dataset"][1],
        split=task["sample_split"],
        streaming=True,
    )
    texts = [
        row[task["text_col"]]
        for row in itertools.islice(ds_stream, max_samples)
        if row.get(task["text_col"])
    ]

    logger.info("Loading model/tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(model, dump_stats_path=str(out_dir / "acta"), tokenizer=tokenizer)
    model.eval()

    losses: List[float] = []
    batches = list(chunked(texts, batch_size=batch_size))
    with torch.no_grad():
        for batch_texts in tqdm(
            batches, desc=f"[text-generation] {model_id}", leave=False
        ):
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(_device())
            outputs = model(**enc, labels=enc["input_ids"])
            losses.append(float(outputs.loss.item()))

    avg_nll = sum(losses) / max(len(losses), 1)
    ppl = math.exp(avg_nll) if avg_nll < 20 else float("inf")
    return {"metric": task["metric"], "score": {"perplexity": ppl, "avg_nll": avg_nll}}


def evaluate_image_classification(
    model_id: str, max_samples: int, batch_size: int, out_dir: Path
) -> Dict:
    task = TASKS["image-classification"]
    logger.info("Loading dataset for image classification: %s", task["dataset"][0])
    ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["sample_split"])
    ds = sample_dataset(ds, SampleConfig(split=task["sample_split"], max_samples=max_samples))

    logger.info("Loading model/feature extractor: %s", model_id)
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(
        model,
        dump_stats_path=str(out_dir / "acta"),
        tokenizer=None,
        vit_reg_patch_labels=True,
    )
    metric = evaluate.load(task["metric"])

    preds: List[int] = []
    refs = ds[task["label_col"]]
    model.eval()
    batches = list(chunked(ds[task["image_col"]], batch_size=batch_size))
    with torch.no_grad():
        for batch in tqdm(
            batches,
            desc=f"[image-classification] {model_id}",
            leave=False,
        ):
            inputs = extractor(images=batch, return_tensors="pt").to(_device())
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    score = metric.compute(predictions=preds, references=refs)
    return {"metric": task["metric"], "score": score}


def evaluate_asr(model_id: str, max_samples: int, batch_size: int, out_dir: Path) -> Dict:
    task = TASKS["automatic-speech-recognition"]
    logger.info("Loading dataset for ASR: %s", task["dataset"][0])
    ds = load_dataset(task["dataset"][0], task["dataset"][1], split=task["sample_split"])
    ds = sample_dataset(ds, SampleConfig(split=task["sample_split"], max_samples=max_samples))
    metric = evaluate.load(task["metric"])

    predictions: List[str] = []
    references = ds[task["text_col"]]

    if "whisper" in model_id.lower():
        logger.info("Loading Whisper-style model: %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(_device())
        model = AutoAnalyzer(model, dump_stats_path=str(out_dir / "acta"), tokenizer=processor.tokenizer, asr_chunk_labels=True)
        model.eval()
        with torch.no_grad():
            for audio in tqdm(
                list(chunked(ds[task["audio_col"]], batch_size=batch_size)),
                desc=f"[automatic-speech-recognition] {model_id}",
                leave=False,
            ):
                for item in audio:
                    inputs = processor(
                        item["array"],
                        sampling_rate=item["sampling_rate"],
                        return_tensors="pt",
                    ).to(_device())
                    generated = model.generate(**inputs, max_new_tokens=96)
                    predictions.append(
                        processor.batch_decode(generated, skip_special_tokens=True)[0]
                    )
    else:
        logger.info("Loading CTC-style model: %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id).to(_device())
        model = AutoAnalyzer(model, dump_stats_path=str(out_dir / "acta"), tokenizer=None, asr_chunk_labels=True)
        model.eval()
        with torch.no_grad():
            for audio in tqdm(
                list(chunked(ds[task["audio_col"]], batch_size=batch_size)),
                desc=f"[automatic-speech-recognition] {model_id}",
                leave=False,
            ):
                inputs = processor(
                    [item["array"] for item in audio],
                    sampling_rate=audio[0]["sampling_rate"],
                    return_tensors="pt",
                    padding=True,
                ).to(_device())
                logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                predictions.extend(processor.batch_decode(pred_ids))

    score = metric.compute(predictions=predictions, references=references)
    return {"metric": task["metric"], "score": score}


EVALUATORS: Dict[str, Callable[[str, int, int, Path], Dict]] = {
    "text-classification": evaluate_text_classification,
    "machine-translation": evaluate_machine_translation,
    "text-generation": evaluate_text_generation,
    "image-classification": evaluate_image_classification,
    "automatic-speech-recognition": evaluate_asr,
}


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Starting domain sweep: max_samples=%d, output_dir=%s", args.max_samples, output_root)
    selected_domains = (
        list(MODEL_GROUPS.keys())
        if "all" in args.tasks
        else [domain for domain in args.tasks if domain in MODEL_GROUPS]
    )
    logger.info("Selected domains: %s", ", ".join(selected_domains))

    report = {"seed": args.seed, "max_samples": args.max_samples, "tasks": selected_domains, "results": {}}
    for domain in selected_domains:
        model_ids = MODEL_GROUPS[domain]
        logger.info("Domain: %s", domain)
        report["results"][domain] = {}
        evaluator = EVALUATORS[domain]
        for model_id in model_ids:
            logger.info("Evaluating model: %s", model_id)
            model_safe = model_id.replace("/", "__")
            run_dir = output_root / domain / model_safe
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = evaluator(model_id, args.max_samples, args.batch_size, run_dir)
                report["results"][domain][model_id] = {"status": "ok", **result}
                logger.info("Completed model: %s", model_id)
            except Exception as exc:
                report["results"][domain][model_id] = {"status": "failed", "error": str(exc)}
                logger.exception("Model failed: %s", model_id)

            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(report["results"][domain][model_id], f, indent=2)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/domain_effect")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", *MODEL_GROUPS.keys()],
        help="Task domains to evaluate (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
