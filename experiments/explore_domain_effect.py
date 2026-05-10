from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import evaluate
import torch
from acta import AutoAnalyzer
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    DataCollatorForLanguageModeling,
    AutoProcessor,
    AutoTokenizer,
)

from experiments.utils import SampleConfig, chunked, sample_dataset, set_seed

logger = logging.getLogger(__name__)


MODEL_GROUPS: Dict[str, List[str]] = {
    "masked-language-modeling": [
        "google-bert/bert-base-uncased",
        "distilbert/distilbert-base-uncased",
        "albert/albert-base-v2",
    ],
    "machine-translation": [
        # "google-t5/t5-small",
        "google/mt5-small",
        "facebook/mbart-large-50-many-to-many-mmt",
    ],
    "text-generation": [
        "gpt2",
        # "google/gemma-2-2b-it",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    "image-classification": [
        "google/vit-base-patch16-224",
        # "google/vit-base-patch32-224",
        "facebook/deit-tiny-patch16-224",
    ],
    "automatic-speech-recognition": [
        "openai/whisper-tiny",
        "facebook/wav2vec2-base-960h",
    ],
}

TARGET_LAYER_PATTERNS = {
    "distilbert/distilbert-base-uncased": [
        "distilbert.transformer.layer.*.attention.q_lin",
        "distilbert.transformer.layer.*.attention.k_lin",
        "distilbert.transformer.layer.*.attention.v_lin",
        "distilbert.transformer.layer.*.attention.out_lin",
        "distilbert.transformer.layer.*.ffn.lin1",
        "distilbert.transformer.layer.*.ffn.lin2",
    ],
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": [
        "distilbert.transformer.layer.*.attention.q_lin",
        "distilbert.transformer.layer.*.attention.k_lin",
        "distilbert.transformer.layer.*.attention.v_lin",
        "distilbert.transformer.layer.*.attention.out_lin",
        "distilbert.transformer.layer.*.ffn.lin1",
        "distilbert.transformer.layer.*.ffn.lin2",
    ],
    "albert/albert-base-v2": [
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.query",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.key",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.value",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.dense",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn_output",
    ],
    "google-bert/bert-base-uncased": [
        "bert.encoder.layer.*.attention.self.query",
        "bert.encoder.layer.*.attention.self.key",
        "bert.encoder.layer.*.attention.self.value",
        "bert.encoder.layer.*.attention.output.dense",
        "bert.encoder.layer.*.intermediate.dense",
        "bert.encoder.layer.*.output.dense",
    ],
    "answerdotai/ModernBERT-base": [
        "model.layers.*.attn.Wqkv",
        "model.layers.*.attn.Wo",
        "model.layers.*.mlp.Wi",
        "model.layers.*.mlp.Wo",
    ],
    "gpt2": [
        "transformer.h.*.attn.c_attn",
        "transformer.h.*.attn.c_proj",
        "transformer.h.*.mlp.c_fc",
        "transformer.h.*.mlp.c_proj",
    ],
    "google/gemma-2-2b-it": [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    ],
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    ],
    "google/vit-base-patch16-224": [
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    "google/vit-base-patch32-224": [
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    "facebook/deit-tiny-patch16-224": [
        "deit.encoder.layer.*.attention.attention.query",
        "deit.encoder.layer.*.attention.attention.key",
        "deit.encoder.layer.*.attention.attention.value",
        "deit.encoder.layer.*.attention.output.dense",
        "deit.encoder.layer.*.intermediate.dense",
        "deit.encoder.layer.*.output.dense",
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    "openai/whisper-tiny": [
        "model.encoder.layers.*.self_attn.q_proj",
        "model.encoder.layers.*.self_attn.k_proj",
        "model.encoder.layers.*.self_attn.v_proj",
        "model.encoder.layers.*.self_attn.out_proj",
        "model.encoder.layers.*.fc1",
        "model.encoder.layers.*.fc2",
    ],
    "facebook/wav2vec2-base-960h": [
        "wav2vec2.encoder.layers.*.attention.q_proj",
        "wav2vec2.encoder.layers.*.attention.k_proj",
        "wav2vec2.encoder.layers.*.attention.v_proj",
        "wav2vec2.encoder.layers.*.attention.out_proj",
        "wav2vec2.encoder.layers.*.feed_forward.intermediate_dense",
        "wav2vec2.encoder.layers.*.feed_forward.output_dense",
    ],
}


TASKS: Dict[str, Dict[str, Any]] = {
    "masked-language-modeling": {
        "dataset": ("wikitext", "wikitext-103-raw-v1"),
        "text_col": "text",
        "sample_split": "validation",
        "metric": "mlm_loss",
        "max_length": 256,
        "mlm_probability": 0.15,
    },
    # "machine-translation": {
    #     "dataset": ("Muennighoff/flores200", "eng_Latn-deu_Latn"),
    #     "src_col": "sentence_eng_Latn",
    #     "tgt_col": "sentence_deu_Latn",
    #     "metric": "sacrebleu",
    #     "sample_split": "dev",
    # },
    "text-generation": {
        "dataset": ("allenai/c4", "en"),
        "text_col": "text",
        "metric": "perplexity",
        "sample_split": "validation",
    },
    "image-classification": {
        "dataset": ("cifar10", None),
        "image_col": "img",
        "label_col": "label",
        "metric": "accuracy",
        "sample_split": "test",
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


def evaluate_masked_language_modeling(
    model_id: str, max_samples: Optional[int], batch_size: int, out_dir: Path
) -> Dict:
    task: Dict[str, Any] = TASKS["masked-language-modeling"]
    split_expr = (
        task["sample_split"]
        if max_samples is None
        else f"{task['sample_split']}[:{max(1, max_samples)}]"
    )
    logger.info("Loading dataset for MLM: %s split=%s", task["dataset"][0], split_expr)
    ds = load_dataset(
        task["dataset"][0],
        task["dataset"][1],
        split=split_expr,
        trust_remote_code=True,
    )

    logger.info("Loading model/tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(
        model,
        dump_stats_path=str(out_dir / "acta"),
        tokenizer=tokenizer,
        target_layers=TARGET_LAYER_PATTERNS[model_id],
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(task["mlm_probability"]),
    )

    texts = [t for t in ds[task["text_col"]] if isinstance(t, str) and t.strip()]
    if len(texts) < len(ds):
        logger.debug("Filtered %d empty wikitext rows", len(ds) - len(texts))

    losses: List[float] = []
    model.eval()
    max_length = int(task["max_length"])
    batches = list(chunked(texts, batch_size=batch_size))
    with torch.no_grad():
        for batch_texts in tqdm(
            batches,
            desc=f"[masked-language-modeling] {model_id}",
            leave=False,
        ):
            features = []
            for text in batch_texts:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                features.append(
                    {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}
                )
            batch = collator(features)
            batch = {k: v.to(_device()) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
            del batch, out

    avg_loss = sum(losses) / max(len(losses), 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {
        "metric": task["metric"],
        "score": {"mlm_loss": avg_loss, "perplexity": ppl},
    }


def evaluate_machine_translation(
    model_id: str, max_samples: Optional[int], batch_size: int, out_dir: Path
) -> Dict:
    task: Dict[str, Any] = TASKS["machine-translation"]
    logger.info("Loading dataset for translation: %s", task["dataset"][0])
    ds = load_dataset(
        task["dataset"][0], task["dataset"][1], split=task["sample_split"], trust_remote_code=True
    )
    ds = sample_dataset(
        ds, SampleConfig(split=task["sample_split"], max_samples=max_samples)
    )

    logger.info("Loading model/tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(
        model, dump_stats_path=str(out_dir / "acta"), tokenizer=tokenizer, target_layers=TARGET_LAYER_PATTERNS[model_id]
    )
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
            enc = tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(_device())
            generated = model.generate(**enc, max_new_tokens=80)
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    score = metric.compute(predictions=preds, references=ref_texts)
    return {"metric": task["metric"], "score": score}


def evaluate_text_generation(
    model_id: str, max_samples: Optional[int], batch_size: int, out_dir: Path
) -> Dict:
    task: Dict[str, Any] = TASKS["text-generation"]
    logger.info("Loading dataset for generation: %s", task["dataset"][0])
    if max_samples is None:
        ds = load_dataset(
            task["dataset"][0],
            task["dataset"][1],
            split=task["sample_split"],
            trust_remote_code=True,
        )
        texts = [row[task["text_col"]] for row in ds if row.get(task["text_col"])]
    else:
        ds_stream = load_dataset(
            task["dataset"][0],
            task["dataset"][1],
            split=task["sample_split"],
            streaming=True,
            trust_remote_code=True,
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
    model = AutoAnalyzer(
        model, dump_stats_path=str(out_dir / "acta"), tokenizer=tokenizer, target_layers=TARGET_LAYER_PATTERNS[model_id]
    )
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
    model_id: str, max_samples: Optional[int], batch_size: int, out_dir: Path
) -> Dict:
    task: Dict[str, Any] = TASKS["image-classification"]
    logger.info("Loading dataset for image classification: %s", task["dataset"][0])
    ds = load_dataset(
        task["dataset"][0], task["dataset"][1], split=task["sample_split"], trust_remote_code=True
    )
    ds = sample_dataset(
        ds, SampleConfig(split=task["sample_split"], max_samples=max_samples)
    )

    logger.info("Loading model/image processor: %s", model_id)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(_device())
    model = AutoAnalyzer(
        model,
        dump_stats_path=str(out_dir / "acta"),
        tokenizer=None,
        vit_reg_patch_labels=True,
        target_layers=TARGET_LAYER_PATTERNS[model_id]
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
            inputs = processor(images=batch, return_tensors="pt").to(_device())
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    score = metric.compute(predictions=preds, references=refs)
    return {"metric": task["metric"], "score": score}


def evaluate_asr(
    model_id: str, max_samples: Optional[int], batch_size: int, out_dir: Path
) -> Dict:
    task: Dict[str, Any] = TASKS["automatic-speech-recognition"]
    logger.info("Loading dataset for ASR: %s", task["dataset"][0])
    ds = load_dataset(
        task["dataset"][0], task["dataset"][1], split=task["sample_split"], trust_remote_code=True
    )
    ds = sample_dataset(
        ds, SampleConfig(split=task["sample_split"], max_samples=max_samples)
    )
    metric = evaluate.load(task["metric"])

    predictions: List[str] = []
    references = ds[task["text_col"]]

    if "whisper" in model_id.lower():
        logger.info("Loading Whisper-style model: %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(_device())
        model = AutoAnalyzer(
            model,
            dump_stats_path=str(out_dir / "acta"),
            tokenizer=processor.tokenizer,
            asr_chunk_labels=True,
            target_layers=TARGET_LAYER_PATTERNS[model_id]
        )
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
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=96,
                        language="en",
                        task="transcribe",
                    )
                    predictions.append(
                        processor.batch_decode(generated, skip_special_tokens=True)[0]
                    )
    else:
        logger.info("Loading CTC-style model: %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id).to(_device())
        model = AutoAnalyzer(
            model,
            dump_stats_path=str(out_dir / "acta"),
            tokenizer=None,
            asr_chunk_labels=True,
            target_layers=TARGET_LAYER_PATTERNS[model_id]
        )
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


EVALUATORS: Dict[str, Callable[[str, Optional[int], int, Path], Dict]] = {
    "masked-language-modeling": evaluate_masked_language_modeling,
    # "machine-translation": evaluate_machine_translation,
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

    batch_size = 1
    if args.batch_size != 1:
        logger.warning("Forcing batch_size=1 (CLI had --batch-size=%d)", args.batch_size)

    logger.info(
        "Starting domain sweep: max_samples=%s, output_dir=%s, batch_size=%d",
        args.max_samples if args.max_samples is not None else "full",
        output_root,
        batch_size,
    )
    selected_domains = (
        list(MODEL_GROUPS.keys())
        if "all" in args.tasks
        else [domain for domain in args.tasks if domain in MODEL_GROUPS]
    )
    logger.info("Selected domains: %s", ", ".join(selected_domains))

    report = {
        "seed": args.seed,
        "max_samples": args.max_samples if args.max_samples is not None else "full",
        "tasks": selected_domains,
        "results": {},
    }
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
                result = evaluator(model_id, args.max_samples, batch_size, run_dir)
                report["results"][domain][model_id] = {"status": "ok", **result}
                logger.info("Completed model: %s", model_id)
            except Exception as exc:
                report["results"][domain][model_id] = {
                    "status": "failed",
                    "error": str(exc),
                }
                logger.exception("Model failed: %s", model_id)

            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(report["results"][domain][model_id], f, indent=2)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
