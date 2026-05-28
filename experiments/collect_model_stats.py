from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset
from experiments import explore_domain_effect as de
from experiments.cache_cleanup import (
    clear_hf_dataset_cache,
    release_memory,
    remove_hf_hub_model_cache,
)
from experiments.utils import SampleConfig, sample_dataset, set_seed

logger = logging.getLogger(__name__)

WIKITEXT_CONFIG = "wikitext-2-raw-v1"

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

DOMAIN_DATASETS: Dict[str, Dict[str, str]] = {
    "text": {"dataset": "wikitext", "config": WIKITEXT_CONFIG},
    "vision": {"dataset": "ILSVRC/imagenet-1k", "config": "(default)"},
    "audio": {"dataset": "librispeech_asr", "config": "clean"},
}

TASK_DATA_ALIASES: Dict[str, str] = {
    "seq2seq": "text-generation",
    "image-generation": "image-classification",
    "image-captioning": "image-classification",
    "speech-representation": "automatic-speech-recognition",
}


@dataclass(frozen=True)
class RunSpec:
    task: str
    model_id: str


def _configure_datasets() -> None:
    for task_key in ("masked-language-modeling", "text-generation"):
        de.TASKS[task_key]["dataset"] = ("wikitext", WIKITEXT_CONFIG)


def _task_data_config(task: str) -> Optional[Dict[str, Any]]:
    if task in de.TASKS:
        return de.TASKS[task]
    alias = TASK_DATA_ALIASES.get(task)
    if alias and alias in de.TASKS:
        return de.TASKS[alias]
    return None


def _load_task_dataset(task: str, max_samples: int, seed: int) -> Dataset:
    task_cfg = _task_data_config(task)
    if task_cfg is None:
        raise KeyError(f"No dataset configuration for task {task!r}")

    data_task = TASK_DATA_ALIASES.get(task, task)
    text_col = task_cfg.get("text_col")
    if text_col and data_task in ("masked-language-modeling", "text-generation", "seq2seq"):
        split_expr = f"{task_cfg['sample_split']}[:{max(1, max_samples)}]"
        ds = load_dataset(
            task_cfg["dataset"][0],
            task_cfg["dataset"][1],
            split=split_expr,
            trust_remote_code=True,
        )
        rows = [
            {text_col: row[text_col]}
            for row in ds
            if isinstance(row.get(text_col), str) and row[text_col].strip()
        ]
        return Dataset.from_list(rows)

    ds = load_dataset(
        task_cfg["dataset"][0],
        task_cfg["dataset"][1],
        split=task_cfg["sample_split"],
        trust_remote_code=True,
    )
    return sample_dataset(
        ds, SampleConfig(split=task_cfg["sample_split"], max_samples=max_samples, seed=seed)
    )


def _dump_text_samples(ds: Dataset, text_col: str, dump_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        records.append({"index": idx, "text": row[text_col]})
    path = dump_dir / "samples.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def _dump_image_samples(
    ds: Dataset, image_col: str, label_col: str, dump_dir: Path
) -> List[Dict[str, Any]]:
    images_dir = dump_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        image_path = images_dir / f"{idx:05d}.png"
        row[image_col].save(image_path)
        records.append(
            {
                "index": idx,
                "label": row[label_col],
                "image_path": str(image_path.relative_to(dump_dir)),
            }
        )
    path = dump_dir / "samples.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def _dump_audio_samples(
    ds: Dataset, audio_col: str, text_col: str, dump_dir: Path
) -> List[Dict[str, Any]]:
    audio_dir = dump_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        audio = row[audio_col]
        wav_path = audio_dir / f"{idx:05d}.wav"
        sf.write(wav_path, np.asarray(audio["array"]), int(audio["sampling_rate"]))
        records.append(
            {
                "index": idx,
                "text": row[text_col],
                "sampling_rate": int(audio["sampling_rate"]),
                "audio_path": str(wav_path.relative_to(dump_dir)),
            }
        )
    path = dump_dir / "samples.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def dump_task_samples(
    task: str, max_samples: int, seed: int, dump_root: Path
) -> Dict[str, Any]:
    task_cfg = _task_data_config(task)
    if task_cfg is None:
        raise KeyError(f"Cannot dump samples for unknown task {task!r}")

    dump_dir = dump_root / "datasets" / task
    dump_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dumping dataset samples for %s (%d max)", task, max_samples)
    ds = _load_task_dataset(task, max_samples=max_samples, seed=seed)

    if "text_col" in task_cfg and "image_col" not in task_cfg:
        records = _dump_text_samples(ds, task_cfg["text_col"], dump_dir)
    elif "image_col" in task_cfg:
        records = _dump_image_samples(ds, task_cfg["image_col"], task_cfg["label_col"], dump_dir)
    elif "audio_col" in task_cfg:
        records = _dump_audio_samples(
            ds, task_cfg["audio_col"], task_cfg["text_col"], dump_dir
        )
    else:
        raise ValueError(f"Unsupported task layout for dumping: {task}")

    meta = {
        "task": task,
        "dataset": task_cfg["dataset"],
        "split": task_cfg.get("sample_split"),
        "num_samples": len(records),
        "dump_dir": str(dump_dir),
    }
    with open(dump_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %d samples to %s", len(records), dump_dir)
    return meta


def _configure_layer_patterns() -> None:
    patterns = de.TARGET_LAYER_PATTERNS
    gpt2 = patterns["gpt2"]
    for model_id in ("gpt2-medium", "gpt2-large"):
        patterns.setdefault(model_id, gpt2)

    whisper = patterns["openai/whisper-tiny"]
    patterns.setdefault("openai/whisper-base", whisper)

    vit = patterns["google/vit-base-patch16-224"]
    patterns.setdefault("google/vit-large-patch16-224", vit)


def _all_specs() -> List[RunSpec]:
    return [RunSpec(task=task, model_id=model_id) for task, model_id in MODEL_SPECS]


def _filter_specs(
    specs: Sequence[RunSpec],
    tasks: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
) -> List[RunSpec]:
    out: List[RunSpec] = []
    for spec in specs:
        if tasks is not None and spec.task not in tasks:
            continue
        if models is not None and spec.model_id not in models:
            continue
        out.append(spec)
    return out


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)
    _configure_datasets()
    _configure_layer_patterns()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    if args.batch_size != 1:
        logger.warning("Forcing batch_size=1 (CLI had --batch-size=%d)", args.batch_size)

    specs = _filter_specs(_all_specs(), args.tasks, args.models)
    if not specs:
        raise SystemExit("No models selected; check --tasks / --models filters.")

    logger.info(
        "Collecting model stats: max_samples=%d, models=%d, output_dir=%s",
        args.max_samples,
        len(specs),
        output_root,
    )

    report: Dict = {
        "seed": args.seed,
        "max_samples": args.max_samples,
        "dump_data": args.dump_data,
        "domain_datasets": DOMAIN_DATASETS,
        "model_specs": [{"task": s.task, "model_id": s.model_id} for s in specs],
        "dataset_dumps": {},
        "results": {},
    }

    datasets_seen: set[tuple[str, Optional[str]]] = set()

    if args.dump_data:
        for task in sorted({spec.task for spec in specs}):
            task_cfg = _task_data_config(task)
            if task_cfg is None:
                logger.warning("Skipping dataset dump for unsupported task: %s", task)
                continue
            try:
                report["dataset_dumps"][task] = dump_task_samples(
                    task,
                    max_samples=args.max_samples,
                    seed=args.seed,
                    dump_root=output_root,
                )
                datasets_seen.add(task_cfg["dataset"])
            except Exception as exc:
                report["dataset_dumps"][task] = {"status": "failed", "error": str(exc)}
                logger.exception("Failed to dump samples for task %s", task)

    for spec in specs:
        if spec.task not in de.EVALUATORS:
            logger.error("No evaluator for task %s; skipping %s", spec.task, spec.model_id)
            report["results"].setdefault(spec.task, {})
            model_safe = spec.model_id.replace("/", "__")
            run_dir = output_root / spec.task / model_safe
            run_dir.mkdir(parents=True, exist_ok=True)
            report["results"][spec.task][spec.model_id] = {
                "status": "skipped",
                "error": f"Unknown task {spec.task!r}",
            }
            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(report["results"][spec.task][spec.model_id], f, indent=2)
            continue

        evaluator = de.EVALUATORS[spec.task]
        task_cfg = _task_data_config(spec.task)
        if task_cfg is None:
            continue
        datasets_seen.add(task_cfg["dataset"])

        logger.info("Running %s on %s", spec.task, spec.model_id)
        model_safe = spec.model_id.replace("/", "__")
        run_dir = output_root / spec.task / model_safe
        run_dir.mkdir(parents=True, exist_ok=True)

        report["results"].setdefault(spec.task, {})
        try:
            # result = evaluator(spec.model_id, args.max_samples, batch_size, run_dir)
            result = {}
            entry = {
                "status": "ok",
                "acta_dir": str(run_dir / "acta"),
                "dataset": task_cfg["dataset"],
                **result,
            }
            dump_meta = report["dataset_dumps"].get(spec.task)
            if args.dump_data and isinstance(dump_meta, dict) and "dump_dir" in dump_meta:
                entry["dataset_dump_dir"] = dump_meta["dump_dir"]
            report["results"][spec.task][spec.model_id] = entry
            logger.info("Completed %s", spec.model_id)
        except Exception as exc:
            report["results"][spec.task][spec.model_id] = {
                "status": "failed",
                "error": str(exc),
                "dataset": task_cfg["dataset"],
            }
            logger.exception("Failed %s", spec.model_id)
        finally:
            remove_hf_hub_model_cache(spec.model_id)
            release_memory()

        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(report["results"][spec.task][spec.model_id], f, indent=2)

    for ds_name, ds_config in datasets_seen:
        clear_hf_dataset_cache(ds_name, ds_config)
    release_memory()

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote summary to %s", output_root / "summary.json")


def parse_args() -> argparse.Namespace:
    task_choices = sorted({task for task, _ in MODEL_SPECS})
    model_choices = [model_id for _, model_id in MODEL_SPECS]

    parser = argparse.ArgumentParser(
        description=()
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Number of examples per model (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Ignored: batch size is always 1 for acta hooks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/model_stats",
        help="Root directory for acta dumps and per-run result.json files.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=task_choices,
        help="Restrict to evaluator task keys (e.g. text-generation). Default: all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=model_choices,
        help="Restrict to specific Hugging Face model ids. Default: all.",
    )
    parser.add_argument(
        "--dump-data",
        action="store_true",
        help=(
            "Save the sampled dataset rows used for inference under "
            "<output-dir>/datasets/<task>/ (JSONL plus images/audio files)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
