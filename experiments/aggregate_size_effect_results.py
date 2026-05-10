from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForSpeechSeq2Seq,
)

logger = logging.getLogger(__name__)

DOMAIN_MODEL_CLASS = {
    "masked-language-modeling": AutoModelForMaskedLM,
    "text-generation": AutoModelForCausalLM,
    "image-classification": AutoModelForImageClassification,
    "automatic-speech-recognition": AutoModelForSpeechSeq2Seq,
}


def _extract_metric_scalar(score: Any, metric_key: str) -> Optional[float]:
    if not isinstance(score, dict):
        return None
    val = score.get(metric_key)
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    if hasattr(val, "item"):
        try:
            x = float(val.item())
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None
    return None


def count_parameters(model_id: str, domain: str) -> int:
    cls = DOMAIN_MODEL_CLASS[domain]
    try:
        model = cls.from_pretrained(model_id, device_map="meta")
    except Exception:
        model = cls.from_pretrained(model_id, low_cpu_mem_usage=True)
    try:
        return int(sum(p.numel() for p in model.parameters()))
    finally:
        del model


def load_result_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def collect_series(
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    aggregated: Dict[str, Any] = {"output_dir": str(output_dir.resolve()), "domains": {}}
    plot_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    if not output_dir.is_dir():
        logger.warning("Output dir missing: %s", output_dir)
        return aggregated, plot_groups

    for domain_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        domain = domain_dir.name
        if domain.startswith(".") or domain.endswith(".json"):
            continue
        aggregated["domains"][domain] = {}
        if domain not in DOMAIN_MODEL_CLASS:
            logger.warning("Unknown domain %s — skipping param counts/plots", domain)
            continue

        for family_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            family = family_dir.name
            aggregated["domains"][domain][family] = {}
            series_rows: List[Dict[str, Any]] = []

            for model_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                model_safe = model_dir.name
                model_id = model_safe.replace("__", "/")

                fp32_path = model_dir / "fp32_result.json"
                q_path = model_dir / "int8_fake_static_result.json"
                fp32 = load_result_json(fp32_path)
                quant = load_result_json(q_path)

                row: Dict[str, Any] = {
                    "model_id": model_id,
                    "paths": {
                        "fp32_result": str(fp32_path),
                        "quantized_result": str(q_path),
                    },
                    "fp32": fp32,
                    "quantized": quant,
                }

                metric_key: Optional[str] = None
                fp32_val: Optional[float] = None
                q_val: Optional[float] = None

                if isinstance(fp32, dict) and fp32.get("status") == "ok":
                    metric_key = str(fp32.get("metric", ""))
                    fp32_val = _extract_metric_scalar(fp32.get("score"), metric_key)

                if isinstance(quant, dict) and quant.get("status") == "ok":
                    mk = str(quant.get("metric", ""))
                    if metric_key and mk != metric_key:
                        logger.warning(
                            "Metric mismatch for %s: fp32=%s quant=%s",
                            model_id,
                            metric_key,
                            mk,
                        )
                    if not metric_key:
                        metric_key = mk
                    q_val = _extract_metric_scalar(quant.get("score"), metric_key or mk)

                row["metric_key"] = metric_key
                row["fp32_metric_value"] = fp32_val
                row["quantized_metric_value"] = q_val

                n_params: Optional[int] = None
                if fp32_val is not None or q_val is not None:
                    try:
                        n_params = count_parameters(model_id, domain)
                    except Exception as exc:
                        logger.warning("Param count failed for %s: %s", model_id, exc)
                row["num_parameters"] = n_params

                aggregated["domains"][domain][family][model_id] = row

                if n_params is not None and metric_key and (
                    fp32_val is not None or q_val is not None
                ):
                    series_rows.append(
                        {
                            "model_id": model_id,
                            "n_params": n_params,
                            "metric_key": metric_key,
                            "fp32_metric": fp32_val,
                            "quant_metric": q_val,
                        }
                    )

            plot_groups[(domain, family)] = series_rows

    return aggregated, plot_groups


def _metric_axis_label(metric_key: str) -> str:
    labels = {
        "mlm_loss": "MLM loss",
        "perplexity": "Perplexity",
        "accuracy": "Accuracy",
        "wer": "WER",
    }
    return labels.get(metric_key, metric_key.replace("_", " ").title())


def plot_family(
    domain: str,
    family: str,
    rows: List[Dict[str, Any]],
    dest_dir: Path,
) -> Optional[Path]:
    usable = [r for r in rows if r.get("fp32_metric") is not None or r.get("quant_metric") is not None]
    if not usable:
        logger.warning("No plottable points for %s / %s", domain, family)
        return None

    usable.sort(key=lambda r: r["n_params"])
    metric_key = usable[0]["metric_key"]
    xs = [r["n_params"] for r in usable]

    ys_fp32: List[float] = []
    ys_q: List[float] = []
    xs_fp32: List[int] = []
    xs_q: List[int] = []
    for r in usable:
        if r.get("fp32_metric") is not None:
            xs_fp32.append(r["n_params"])
            ys_fp32.append(float(r["fp32_metric"]))
        if r.get("quant_metric") is not None:
            xs_q.append(r["n_params"])
            ys_q.append(float(r["quant_metric"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    if xs_fp32:
        ax.plot(
            xs_fp32,
            ys_fp32,
            marker="o",
            linestyle="-",
            label="Full precision",
            color="#1f77b4",
        )
    if xs_q:
        ax.plot(
            xs_q,
            ys_q,
            marker="s",
            linestyle="--",
            label="Quantized (INT8)",
            color="#ff7f0e",
        )

    ax.set_xlabel("Parameters")
    ax.set_ylabel(_metric_axis_label(metric_key))
    ax.set_title(f"{family} — {domain}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "metric_vs_params.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote chart %s", out_path)
    return out_path


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    root = Path(args.output_dir).resolve()
    aggregated, plot_groups = collect_series(root)

    chart_paths: Dict[str, str] = {}
    for (domain, family), rows in plot_groups.items():
        path = plot_family(domain, family, rows, root / domain / family)
        if path is not None:
            chart_paths[f"{domain}/{family}"] = str(path)

    aggregated["charts"] = chart_paths

    agg_path = root / "aggregated_results.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    logger.info("Wrote %s", agg_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/size_effect",
        help="Root dir containing domain/family/model/__ results.",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
