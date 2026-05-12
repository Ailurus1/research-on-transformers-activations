from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FORMAT_CHOICES: tuple[str, ...] = (
    "fp16",
    "bf16",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
    "int4",
    "nvfp16",
    "nvfp4",
)

DEFAULT_FORMAT_ORDER: tuple[str, ...] = FORMAT_CHOICES


def all_fp16_finite() -> np.ndarray:
    u = np.arange(65536, dtype=np.uint16)
    v = u.view(np.float16).astype(np.float64)
    v = v[np.isfinite(v)]
    return np.unique(v)


def all_bfloat16_finite() -> np.ndarray:
    u = np.arange(65536, dtype=np.uint32)
    v = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    v = v[np.isfinite(v)]
    return np.unique(v)


def all_fp8_e4m3_finite() -> np.ndarray:
    import torch

    vals: List[float] = []
    for u in range(256):
        b = torch.tensor([u], dtype=torch.uint8)
        t = b.view(torch.float8_e4m3fn)
        x = t.float().numpy().astype(np.float64)[0]
        if np.isfinite(x):
            vals.append(float(x))
    return np.unique(np.array(vals, dtype=np.float64))


def all_fp8_e5m2_finite() -> np.ndarray:
    import torch

    vals: List[float] = []
    for u in range(256):
        b = torch.tensor([u], dtype=torch.uint8)
        t = b.view(torch.float8_e5m2)
        x = t.float().numpy().astype(np.float64)[0]
        if np.isfinite(x):
            vals.append(float(x))
    return np.unique(np.array(vals, dtype=np.float64))


def all_int8() -> np.ndarray:
    return np.arange(-128, 128, dtype=np.int16).astype(np.float64)


def all_int4_signed() -> np.ndarray:
    return np.arange(-8, 8, dtype=np.int16).astype(np.float64)


def all_fp4_e2m1_finite() -> np.ndarray:
    vals: List[float] = []
    for b in range(16):
        s = (b >> 3) & 1
        e = (b >> 1) & 3
        m = b & 1
        bias = 1
        sign = -1.0 if s else 1.0
        if e == 0 and m == 0:
            v = 0.0
        elif e == 0:
            v = sign * (2.0 ** (1 - bias)) * (m / 2.0)
        elif e == 3 and m == 1:
            continue  # treat as NaN, skip
        elif e == 3 and m == 0:
            continue  # omit ±Inf on axis
        else:
            v = sign * (1.0 + m / 2.0) * (2.0 ** (e - bias))
        if math.isfinite(v):
            vals.append(v)
    return np.unique(np.array(vals, dtype=np.float64))


FORMAT_DISPLAY: dict[str, str] = {
    "fp16": "fp16",
    "bf16": "bf16",
    "fp8_e4m3": "fp8 e4m3",
    "fp8_e5m2": "fp8 e5m2",
    "int8": "int8",
    "int4": "int4",
    "nvfp16": "nvfp16",
    "nvfp4": "nvfp4 (FP4 E2M1)",
}


def build_plot_title(order: list[str]) -> str:
    names = ", ".join(FORMAT_DISPLAY[k] for k in order)
    return f"Representable finite values (x ≠ 0; one point per value)\n{names}"


def build_series(keys: List[str]) -> Dict[str, np.ndarray]:
    series: Dict[str, np.ndarray] = {}
    fp16_v: np.ndarray | None = None

    want_fp16 = "fp16" in keys
    want_nvfp16 = "nvfp16" in keys
    if want_fp16 or want_nvfp16:
        fp16_v = all_fp16_finite()
        if want_fp16:
            series["fp16"] = fp16_v
        if want_nvfp16:
            series["nvfp16"] = fp16_v.copy()

    if "bf16" in keys:
        series["bf16"] = all_bfloat16_finite()
    if "fp8_e4m3" in keys:
        series["fp8_e4m3"] = all_fp8_e4m3_finite()
    if "fp8_e5m2" in keys:
        series["fp8_e5m2"] = all_fp8_e5m2_finite()
    if "int8" in keys:
        series["int8"] = all_int8()
    if "int4" in keys:
        series["int4"] = all_int4_signed()
    if "nvfp4" in keys:
        series["nvfp4"] = all_fp4_e2m1_finite()

    return series


def plot_value_sets(
    series: Dict[str, np.ndarray],
    order: List[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    y_step = 1.0
    for i, key in enumerate(order):
        xv = np.asarray(series[key], dtype=np.float64)
        xv = xv[np.isfinite(xv)]
        xv = xv[xv != 0]
        if xv.size == 0:
            continue
        yv = np.full(xv.shape, i * y_step, dtype=np.float64)
        ax.scatter(
            xv,
            yv,
            s=14.0,
            alpha=0.88,
            linewidths=0.35,
            edgecolors="0.15",
            label=f"{FORMAT_DISPLAY[key]} (n={len(xv)})",
        )

    ax.set_yticks([i * y_step for i in range(len(order))])
    ax.set_yticklabels([FORMAT_DISPLAY[k] for k in order])
    ax.set_xlabel("numeric value (linear)")
    ax.set_title(title)
    ax.set_xscale("linear")
    ax.margins(x=0.02)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "datatype_distributions.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=FORMAT_CHOICES,
        metavar="FMT",
        help=(
            "Dtype to include in the plot (repeatable). "
            f"Choices: {', '.join(FORMAT_CHOICES)}. "
            "If omitted, all types are plotted."
        ),
    )
    args = parser.parse_args()

    order = list(args.formats) if args.formats else list(DEFAULT_FORMAT_ORDER)
    seen: set[str] = set()
    order_unique: List[str] = []
    for k in order:
        if k not in seen:
            seen.add(k)
            order_unique.append(k)
    order = order_unique

    series = build_series(order)
    for k in order:
        if k not in series:
            raise KeyError(k)

    title = build_plot_title(order)
    plot_value_sets(series, order, args.out, title)
    print(f"Wrote {args.out} ({', '.join(order)})")


if __name__ == "__main__":
    main()
