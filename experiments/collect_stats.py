from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

OUTLIERS_DETECTED_COL = "outliers detected"
TOKEN_COL = "token"
IDX_COL = "idx"

PUNCTUATION_CHARS = frozenset({",", ".", "!", "?", ";", ":", "“", "”", "’", "-", "--", "---", "\n", "\r"})

FUNCTION_WORDS = frozenset({
    "in",
    "on",
    "at",
    "to",
    "with",
    "and",
    "but",
    "or",
    "because",
    "a",
    "an",
    "the",
    "not",
    "up",
    "down",
})

CATEGORIES: tuple[str, ...] = (
    "Initial token",
    "Special tokens",
    "Punctuation marks",
    "First token after punctuation mark",
    "Function words",
    "Other tokens",
)

CATEGORY_COLORS: dict[str, str] = {
    "Initial token": "#4477AA",
    "Special tokens": "#EE6677",
    "Punctuation marks": "#228833",
    "First token after punctuation mark": "#CCBB44",
    "Function words": "#66CCEE",
    "Other tokens": "#AA3377",
}

_SPECIAL_BRACKET = re.compile(r"^\[[^\]]+\]$")
_SPECIAL_ANGLE = re.compile(r"^<[^>]+>$")


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_model_special_tokens(run_dir: Path) -> set[str]:
    stats_path = run_dir / "stats.json"
    if not stats_path.is_file():
        return set()
    try:
        with open(stats_path, encoding="utf-8") as f:
            meta = json.load(f).get("_acta", {})
        model_name = meta.get("model_name")
        if not model_name:
            return set()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_name),
            use_fast=("deberta" not in str(model_name).lower()),
        )
        specials: set[str] = set()
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ):
            tok = getattr(tokenizer, key, None)
            if isinstance(tok, str) and tok:
                specials.add(tok)
        if hasattr(tokenizer, "all_special_tokens"):
            specials.update(str(t) for t in tokenizer.all_special_tokens)
        if hasattr(tokenizer, "additional_special_tokens"):
            specials.update(
                str(t) for t in (tokenizer.additional_special_tokens or [])
            )
        return specials
    except Exception:
        return set()


def _is_heuristic_special_token(token: str) -> bool:
    text = token.strip()
    if not text:
        return False
    if _SPECIAL_BRACKET.match(text) or _SPECIAL_ANGLE.match(text):
        return True
    lowered = text.lower()
    if "redacted" in lowered:
        return True
    if text in ("<|endoftext|>", "<|startoftext|>"):
        return True
    return False


def _is_special_token(token: str, model_specials: set[str]) -> bool:
    if token in model_specials:
        return True
    stripped = token.strip()
    if stripped in model_specials:
        return True
    return _is_heuristic_special_token(token)


def _is_punctuation_token(token: str) -> bool:
    text = token.strip()
    if not text:
        return False
    if text in PUNCTUATION_CHARS:
        return True
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    return all(c in PUNCTUATION_CHARS for c in non_space)


def _is_function_word(token: str) -> bool:
    text = token.strip().lower()
    return bool(text) and text in FUNCTION_WORDS


def classify_token(
    idx: int,
    token: str,
    prev_token: str | None,
    *,
    model_specials: set[str],
) -> str:
    if idx == 0:
        return CATEGORIES[0]
    if _is_special_token(token, model_specials):
        return CATEGORIES[1]
    if _is_punctuation_token(token):
        return CATEGORIES[2]
    if prev_token is not None and _is_punctuation_token(prev_token):
        return CATEGORIES[3]
    if _is_function_word(token):
        return CATEGORIES[4]
    return CATEGORIES[5]


def _discover_sequence_csvs(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file() and path.name.startswith("acta_results_") and path.suffix == ".csv":
        return [path]
    seq_dir = path / "sequences" if path.is_dir() else None
    if seq_dir is not None and seq_dir.is_dir():
        files = sorted(seq_dir.glob("acta_results_*.csv"))
        if files:
            return files
    if path.is_dir():
        files = sorted(path.rglob("sequences/acta_results_*.csv"))
        if files:
            return files
    raise FileNotFoundError(
        f"No per-sequence CSV files under {path} "
        "(expected sequences/acta_results_*.csv)."
    )


def _resolve_run_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "sequences").is_dir() or path.name.startswith("acta_results_"):
        return path if path.is_dir() else path.parent.parent
    if path.is_dir():
        runs = sorted(
            [p for p in path.iterdir() if p.is_dir() and (p / "sequences").is_dir()],
            key=lambda p: p.name,
        )
        if len(runs) == 1:
            return runs[0]
        if runs:
            return runs[-1]
    raise FileNotFoundError(f"Could not resolve acta run directory from {path}")


def load_sequence_rows(csv_path: Path) -> list[tuple[int, str, bool]]:
    rows: list[tuple[int, str, bool]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        for row in reader:
            try:
                idx = int(row.get(IDX_COL, -1))
            except (TypeError, ValueError):
                continue
            token = str(row.get(TOKEN_COL, ""))
            outlier = _parse_bool(row.get(OUTLIERS_DETECTED_COL))
            rows.append((idx, token, outlier))
    rows.sort(key=lambda r: r[0])
    return rows


def collect_category_counts(
    csv_files: Iterable[Path],
    *,
    model_specials: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for csv_path in csv_files:
        seq = load_sequence_rows(csv_path)
        for i, (idx, token, is_outlier) in enumerate(seq):
            if not is_outlier:
                continue
            prev_token = seq[i - 1][1] if i > 0 else None
            category = classify_token(
                idx, token, prev_token, model_specials=model_specials
            )
            counts[category] += 1
    return counts


def plot_pie_chart(
    counts: Counter[str],
    *,
    output_path: Path,
    title: str,
) -> None:
    labels: list[str] = []
    sizes: list[int] = []
    colors: list[str] = []
    for cat in CATEGORIES:
        n = int(counts.get(cat, 0))
        if n > 0:
            labels.append(cat)
            sizes.append(n)
            colors.append(CATEGORY_COLORS[cat])

    if not sizes:
        print("No outlier tokens found; skipping pie chart.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title(title, pad=24)
    ax.axis("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote pie chart ({sum(sizes)} outlier tokens) to {output_path}")


def print_summary(counts: Counter[str], *, csv_count: int) -> None:
    total = sum(counts.values())
    print(f"Per-sequence CSV files: {csv_count}")
    print(f"Outlier tokens (total): {total}")
    for cat in CATEGORIES:
        n = int(counts.get(cat, 0))
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {cat}: {n} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pie chart of outlier-token categories from acta per-sequence CSVs."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(".acta_dump_results"),
        help="Acta run directory, dump root, or a single sequences/acta_results_*.csv file "
        "(default: .acta_dump_results, uses latest run if needed).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <run_dir>/outlier_token_categories_pie.png).",
    )
    args = parser.parse_args()

    csv_files = _discover_sequence_csvs(args.path)
    run_dir = _resolve_run_dir(args.path)
    model_specials = _load_model_special_tokens(run_dir)

    counts = collect_category_counts(csv_files, model_specials=model_specials)
    print_summary(counts, csv_count=len(csv_files))

    out_path = args.output or (run_dir / "outlier_token_categories_pie.png")
    title = f"Outlier tokens by category ({run_dir.name})"
    plot_pie_chart(counts, output_path=out_path, title=title)


if __name__ == "__main__":
    main()
