from __future__ import annotations

import io
from typing import Any

NUM_AUDIO_SAMPLES = 32

LIBRISPEECH_DATASET = "librispeech_asr"
LIBRISPEECH_CONFIG = "clean"
LIBRISPEECH_SPLIT = "validation"
AUDIO_SAMPLING_RATE = 16_000


def load_audio_samples(n: int = NUM_AUDIO_SAMPLES) -> list[dict[str, Any]]:
    """Load ``n`` LibriSpeech validation clips as ``{array, sampling_rate}`` dicts."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "This module requires the `datasets` package. Install with: pip install datasets"
        ) from exc

    kwargs: dict[str, Any] = {"split": LIBRISPEECH_SPLIT, "streaming": True}
    stream = load_dataset(LIBRISPEECH_DATASET, LIBRISPEECH_CONFIG, **kwargs)
    stream = stream.cast_column(
        "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE, decode=False)
    )

    clips: list[dict[str, Any]] = []
    for row in stream:
        audio = row.get("audio")
        if not isinstance(audio, dict):
            continue
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if array is not None and sampling_rate is not None:
            clips.append(
                {
                    "array": array,
                    "sampling_rate": int(sampling_rate),
                }
            )
        elif audio.get("bytes"):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError(
                    "Install soundfile for LibriSpeech without torchcodec: "
                    "pip install soundfile"
                ) from exc
            data, sr = sf.read(io.BytesIO(audio["bytes"]))
            clips.append({"array": data, "sampling_rate": int(sr)})
        elif audio.get("path"):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError(
                    "Install soundfile for LibriSpeech without torchcodec: "
                    "pip install soundfile"
                ) from exc
            data, sr = sf.read(audio["path"])
            clips.append({"array": data, "sampling_rate": int(sr)})
        else:
            continue
        if len(clips) >= n:
            break
    if len(clips) < n:
        raise RuntimeError(f"Only loaded {len(clips)} audio clips (need {n})")
    return clips


assert NUM_AUDIO_SAMPLES == 32
