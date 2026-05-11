from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def remove_hf_hub_model_cache(repo_id: str) -> None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    if not hub.is_dir():
        return
    token = "models--" + repo_id.replace("/", "--")
    target = hub / token
    if target.is_dir():
        try:
            shutil.rmtree(target, ignore_errors=True)
            logger.info("Removed HF hub cache for model: %s", repo_id)
        except OSError as exc:
            logger.warning("Could not remove hub cache %s: %s", target, exc)


def clear_hf_dataset_cache(dataset_path: str, config: Optional[str] = None) -> None:
    try:
        from datasets import load_dataset_builder
    except ImportError:
        return

    try:
        builder = load_dataset_builder(
            dataset_path,
            config,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve dataset builder for cache clear (%s, %s): %s",
            dataset_path,
            config,
            exc,
        )
        return

    cache_dir = getattr(builder, "cache_dir", None)
    if cache_dir:
        p = Path(cache_dir)
        if p.is_dir():
            try:
                shutil.rmtree(p, ignore_errors=True)
                logger.info("Removed HF datasets cache: %s", p)
            except OSError as exc:
                logger.warning("Could not remove dataset cache %s: %s", p, exc)

    release_memory()
