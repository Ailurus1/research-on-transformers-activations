from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from experiments.explore_size_effect import (
    FakeQuantLinear,
    _fake_quantize_affine,
    _replace_linear_with_fake_quant,
)
from experiments.gpt2_train_mods import (
    apply_attention_bias,
    apply_context_aware_scaling,
    apply_op_blocks,
    disable_qat_training,
    enable_qat_training,
    set_attention_linear_bias,
)
from experiments.utils import set_seed

logger = logging.getLogger(__name__)

WIKITEXT_CONFIG = "wikitext-2-raw-v1"
SUPPORTED_GPT2_MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


@dataclass
class TrainOptions:
    model_name: str
    optimizer: str
    attention_bias: bool
    attention_linear_bias: bool
    context_aware_scaling: bool
    op_blocks: bool
    qat: bool
    label_smoothing: float

    def run_name(self) -> str:
        parts = [
            self.model_name.replace("-", ""),
            "wt2",
            "soap" if self.optimizer == "soap" else "adamw",
            "attnbias" if self.attention_bias else "noattnbias",
            "linbias" if self.attention_linear_bias else "nolinbias",
            "cascale" if self.context_aware_scaling else "nocascale",
            "opblock" if self.op_blocks else "noopblock",
            "qat" if self.qat else "noqat",
        ]
        if self.label_smoothing > 0:
            parts.append(f"ls{self.label_smoothing:g}".replace(".", "p"))
        else:
            parts.append("nols")
        return "_".join(parts)


def build_train_options(args: argparse.Namespace) -> TrainOptions:
    return TrainOptions(
        model_name=args.model_name,
        optimizer=args.optimizer,
        attention_bias=args.attention_bias,
        attention_linear_bias=args.attention_linear_bias,
        context_aware_scaling=args.context_aware_scaling,
        op_blocks=args.op_blocks,
        qat=args.qat,
        label_smoothing=args.label_smoothing,
    )


def build_model(opts: TrainOptions) -> GPT2LMHeadModel:
    config = GPT2Config.from_pretrained(opts.model_name)
    model = GPT2LMHeadModel(config)

    set_attention_linear_bias(model, opts.attention_linear_bias)
    if opts.attention_bias:
        model = apply_attention_bias(model)
    if opts.context_aware_scaling:
        model = apply_context_aware_scaling(model)
    if opts.op_blocks:
        if opts.attention_bias or opts.context_aware_scaling:
            logger.warning(
                "OP blocks replace standard GPT-2 blocks; systematic attention mods are ignored."
            )
        model = apply_op_blocks(model)

    if opts.qat:
        enable_qat_training(model)

    return model


def tokenize_and_group(
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    block_size: int,
) -> Dict[str, List[List[int]]]:
    ids: List[int] = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])

    if len(ids) < block_size:
        return {"input_ids": [], "attention_mask": []}

    n_blocks = len(ids) // block_size
    blocks = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    return {
        "input_ids": blocks,
        "attention_mask": [[1] * block_size for _ in blocks],
    }


def load_wikitext_blocks(
    split: str,
    tokenizer: GPT2Tokenizer,
    block_size: int,
) -> Dict[str, List[List[int]]]:
    ds = load_dataset("wikitext", WIKITEXT_CONFIG, split=split)
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    return tokenize_and_group(tokenizer, texts, block_size)


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]]) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


def _training_args_kwargs(base: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(TrainingArguments.__init__).parameters
    out = dict(base)
    renames = {"evaluation_strategy": "eval_strategy"}
    for old, new in renames.items():
        if old in out and old not in params and new in params:
            out[new] = out.pop(old)
        elif new in out and new not in params and old in params:
            out[old] = out.pop(new)
    return {k: v for k, v in out.items() if k in params}


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if args.optimizer == "soap":
        try:
            from pytorch_optimizer import SOAP
        except ImportError as exc:
            raise ImportError(
                "SOAP optimizer requires pytorch-optimizer: pip install pytorch-optimizer"
            ) from exc
        return SOAP(
            groups,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            precondition_frequency=args.soap_precondition_frequency,
        )

    import inspect

    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra = {"fused": True} if fused and torch.cuda.is_available() else {}
    return torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        **extra,
    )


@dataclass
class EpochHistory:
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    eval_perplexity: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "eval_perplexity": self.eval_perplexity,
        }


class EpochMetricsCallback(TrainerCallback):
    def __init__(self, run_dir: Path, run_name: str) -> None:
        self.run_dir = run_dir
        self.run_name = run_name
        self.history = EpochHistory()

    def _mean_train_loss_for_epoch(self, state: Any, epoch: int) -> Optional[float]:
        losses = [
            float(log["loss"])
            for log in state.log_history
            if "loss" in log and int(log.get("epoch", -1)) == epoch
        ]
        if not losses:
            return None
        return sum(losses) / len(losses)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None or "eval_loss" not in metrics:
            return
        epoch = max(1, int(round(state.epoch)))
        if self.history.epoch and self.history.epoch[-1] == epoch:
            return

        eval_loss = float(metrics["eval_loss"])
        eval_ppl = float(metrics.get("eval_perplexity", math.exp(min(eval_loss, 80))))

        train_loss = self._mean_train_loss_for_epoch(state, epoch)
        if train_loss is None:
            train_loss = float(metrics.get("train_loss", eval_loss))

        self.history.epoch.append(epoch)
        self.history.train_loss.append(train_loss)
        self.history.eval_loss.append(eval_loss)
        self.history.eval_perplexity.append(eval_ppl)

        logger.info(
            "Epoch %d — train_loss=%.4f eval_loss=%.4f eval_ppl=%.2f",
            epoch,
            train_loss,
            eval_loss,
            eval_ppl,
        )

    def on_train_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if int(os.environ.get("LOCAL_RANK", -1)) not in (-1, 0):
            return
        if not self.history.epoch:
            return
        save_training_curves(self.history, self.run_dir, self.run_name)


def save_training_curves(history: EpochHistory, run_dir: Path, run_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.epoch

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, history.train_loss, "o-", label="train loss", color="tab:blue")
    ax_loss.plot(epochs, history.eval_loss, "s-", label="eval loss", color="tab:orange")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss per epoch")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_ppl.plot(epochs, history.eval_perplexity, "D-", label="eval perplexity", color="tab:green")
    ax_ppl.set_xlabel("epoch")
    ax_ppl.set_ylabel("perplexity")
    ax_ppl.set_title("Validation perplexity per epoch")
    ax_ppl.legend()
    ax_ppl.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = run_dir / f"{run_name}_training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    history_path = run_dir / f"{run_name}_epoch_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.as_dict(), f, indent=2)

    logger.info("Saved training curves to %s", plot_path)


def _history_from_log_history(log_history: List[Dict[str, Any]]) -> EpochHistory:
    history = EpochHistory()
    by_epoch: Dict[int, Dict[str, float]] = {}
    for rec in log_history:
        epoch_raw = rec.get("epoch")
        if epoch_raw is None:
            continue
        epoch = max(1, int(round(float(epoch_raw))))
        slot = by_epoch.setdefault(epoch, {})
        if "loss" in rec:
            slot.setdefault("train_loss_sum", 0.0)
            slot.setdefault("train_loss_n", 0.0)
            slot["train_loss_sum"] += float(rec["loss"])
            slot["train_loss_n"] += 1.0
        if "eval_loss" in rec:
            slot["eval_loss"] = float(rec["eval_loss"])
        if "eval_perplexity" in rec:
            slot["eval_perplexity"] = float(rec["eval_perplexity"])

    for epoch in sorted(by_epoch.keys()):
        slot = by_epoch[epoch]
        if "eval_loss" not in slot:
            continue
        train_n = max(1.0, slot.get("train_loss_n", 0.0))
        train_loss = slot.get("train_loss_sum", slot["eval_loss"]) / train_n
        eval_loss = slot["eval_loss"]
        eval_ppl = slot.get("eval_perplexity", math.exp(min(eval_loss, 80.0)))
        history.epoch.append(epoch)
        history.train_loss.append(float(train_loss))
        history.eval_loss.append(float(eval_loss))
        history.eval_perplexity.append(float(eval_ppl))
    return history


def ensure_training_curves_saved(
    callback: EpochMetricsCallback, log_history: List[Dict[str, Any]], run_dir: Path, run_name: str
) -> None:
    if int(os.environ.get("LOCAL_RANK", -1)) not in (-1, 0):
        return
    history = callback.history if callback.history.epoch else _history_from_log_history(log_history)
    if not history.epoch:
        logger.warning("No epoch metrics available; skipping training curve save.")
        return
    save_training_curves(history, run_dir, run_name)


class TrainParamsTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_options: TrainOptions,
        cli_args: argparse.Namespace,
        eval_tokenizer: GPT2Tokenizer,
        **kwargs,
    ) -> None:
        self.train_options = train_options
        self.cli_args = cli_args
        self.eval_tokenizer = eval_tokenizer
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.model, self.cli_args)
        return self.optimizer

    def _unwrap_model(self) -> nn.Module:
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def _eval_max_length(self) -> int:
        model = self._unwrap_model()
        return min(self.cli_args.block_size, int(model.config.n_positions))

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **kwargs: Any,
    ):
        # Use the same sliding-window recipe as final test eval (not fixed-block CE).
        model = self._unwrap_model()
        device = self.args.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        split = "validation" if metric_key_prefix == "eval" else "test"
        result = eval_wikitext2_perplexity(
            model,
            self.eval_tokenizer,
            device,
            max_length=self._eval_max_length(),
            stride=self.cli_args.eval_stride,
            split=split,
        )
        return {
            f"{metric_key_prefix}_loss": result["avg_nll"],
            f"{metric_key_prefix}_perplexity": result["perplexity"],
            f"{metric_key_prefix}_tokens_scored": result["tokens_scored"],
        }


@torch.no_grad()
def eval_wikitext2_perplexity(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_length: int,
    stride: int,
    split: str = "test",
) -> Dict[str, float]:
    n_positions = int(getattr(model.config, "n_positions", 1024))
    max_length = min(max_length, n_positions)
    stride = max(1, min(stride, max_length))

    ds = load_dataset("wikitext", WIKITEXT_CONFIG, split=split)
    text = "\n\n".join(t for t in ds["text"] if isinstance(t, str) and t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        raise RuntimeError(f"WikiText-2 {split} split is too short for perplexity eval.")

    model.eval()
    nll_sum = 0.0
    prev_end_loc = 0
    n_windows = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc=f"ppl-{split}", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break

        chunk = input_ids[:, begin_loc:end_loc]
        labels = chunk.clone()
        labels[:, :-trg_len] = -100

        outputs = model(chunk, labels=labels)
        loss = outputs.loss
        if loss is None or not math.isfinite(float(loss)):
            raise RuntimeError(f"Non-finite loss in {split} perplexity window at {begin_loc}:{end_loc}")

        nll_sum += float(loss.item()) * trg_len
        n_windows += 1
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    if prev_end_loc == 0:
        raise RuntimeError(f"No perplexity windows for WikiText-2 {split} split.")

    avg_nll = nll_sum / prev_end_loc
    perplexity = math.exp(avg_nll) if avg_nll < 80 else float("inf")
    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "tokens_scored": prev_end_loc,
        "seq_len": seq_len,
        "max_length": max_length,
        "stride": stride,
        "n_windows": n_windows,
    }


def apply_int8_fake_quant(model: nn.Module) -> nn.Module:
    qmodel = copy.deepcopy(model)
    _replace_linear_with_fake_quant(qmodel)

    def _patch_conv1d(module: nn.Module) -> None:
        from transformers.pytorch_utils import Conv1D

        for name, child in list(module.named_children()):
            if isinstance(child, Conv1D):
                linear = nn.Linear(child.weight.shape[0], child.weight.shape[1], bias=child.bias is not None)
                linear.weight = nn.Parameter(child.weight.T.contiguous())
                if child.bias is not None:
                    linear.bias = child.bias
                fq = FakeQuantLinear(linear)
                fq.q_weight = nn.Parameter(_fake_quantize_affine(linear.weight.detach().float()))
                if linear.bias is not None:
                    fq.bias = linear.bias.detach().float()
                setattr(module, name, _FakeQuantConv1D(fq))
            else:
                _patch_conv1d(child)

    _patch_conv1d(qmodel)
    return qmodel


class _FakeQuantConv1D(nn.Module):
    def __init__(self, fq: FakeQuantLinear) -> None:
        super().__init__()
        self.fq = fq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fq(x)
        size_out = x.size()[:-1] + (self.fq.q_weight.shape[0],)
        return out.view(size_out)


def save_run_artifacts(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    run_dir: Path,
    opts: TrainOptions,
    metrics: Dict[str, Any],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"{opts.run_name()}.pt"
    tokenizer_path = run_dir / f"{opts.run_name()}_tokenizer"
    hf_dir = run_dir / f"{opts.run_name()}_hf"

    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, model_path)
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(tokenizer_path)

    with open(run_dir / f"{opts.run_name()}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved checkpoint to %s", model_path)
    return model_path


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    opts = build_train_options(args)
    run_name = opts.run_name()
    run_dir = Path(args.output_dir) / run_name

    set_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(opts.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_blocks = load_wikitext_blocks("train", tokenizer, args.block_size)
    if not train_blocks["input_ids"]:
        raise RuntimeError("WikiText-2 train split produced zero blocks; check block_size.")

    train_ds = BlockDataset(train_blocks["input_ids"], train_blocks["attention_mask"])
    # Trainer needs a non-empty eval_dataset to trigger evaluate(); metrics use sliding-window PPL.
    eval_ds = train_ds
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    metrics_callback = EpochMetricsCallback(run_dir, run_name)

    model = build_model(opts)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_ddp = n_gpus > 1 and not args.single_gpu
    world_size = n_gpus if use_ddp else 1
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_ds)
            / (
                args.per_device_train_batch_size
                * world_size
                * args.gradient_accumulation_steps
            )
        ),
    )
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else int(steps_per_epoch * args.num_train_epochs)
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    eval_strategy = "epoch" if args.num_train_epochs > 0 else "no"
    training_args = TrainingArguments(
        **_training_args_kwargs(
            {
                "output_dir": str(run_dir / "checkpoints"),
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps if args.max_steps > 0 else -1,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": "cosine",
                "logging_steps": args.logging_steps,
                "logging_strategy": "steps",
                "save_strategy": "no",
                "evaluation_strategy": eval_strategy,
                "do_eval": True,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "report_to": "none",
                "seed": args.seed,
                "dataloader_num_workers": args.dataloader_num_workers,
                "fp16": args.fp16 and torch.cuda.is_available(),
                "bf16": args.bf16 and torch.cuda.is_available(),
                "label_smoothing_factor": opts.label_smoothing,
                **({"ddp_find_unused_parameters": False} if use_ddp else {}),
            }
        )
    )

    trainer = TrainParamsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        train_options=opts,
        cli_args=args,
        eval_tokenizer=tokenizer,
        callbacks=[metrics_callback],
    )

    logger.info("Training run=%s on %d block(s), ddp=%s", run_name, len(train_ds), use_ddp)
    trainer.train()
    ensure_training_curves_saved(metrics_callback, trainer.state.log_history, run_dir, run_name)
    if opts.qat:
        disable_qat_training()

    trained_model = trainer.model
    if hasattr(trained_model, "module"):
        trained_model = trained_model.module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)

    eval_max_length = min(args.block_size, trained_model.config.n_positions)
    fp32_val_metrics = eval_wikitext2_perplexity(
        trained_model,
        tokenizer,
        device,
        max_length=eval_max_length,
        stride=args.eval_stride,
        split="validation",
    )
    logger.info("FP32 validation perplexity: %.4f", fp32_val_metrics["perplexity"])

    fp32_metrics = eval_wikitext2_perplexity(
        trained_model,
        tokenizer,
        device,
        max_length=eval_max_length,
        stride=args.eval_stride,
        split="test",
    )
    logger.info("FP32 test perplexity: %.4f", fp32_metrics["perplexity"])

    int8_model = apply_int8_fake_quant(trained_model).to(device)
    int8_metrics = eval_wikitext2_perplexity(
        int8_model,
        tokenizer,
        device,
        max_length=eval_max_length,
        stride=args.eval_stride,
        split="test",
    )
    logger.info("INT8 fake-quant test perplexity: %.4f", int8_metrics["perplexity"])

    metrics = {
        "run_name": run_name,
        "epoch_history": metrics_callback.history.as_dict(),
        "options": {
            "optimizer": opts.optimizer,
            "attention_bias": opts.attention_bias,
            "attention_linear_bias": opts.attention_linear_bias,
            "context_aware_scaling": opts.context_aware_scaling,
            "op_blocks": opts.op_blocks,
            "qat": opts.qat,
            "label_smoothing": opts.label_smoothing,
        },
        "fp32_validation": fp32_val_metrics,
        "fp32_test": fp32_metrics,
        "int8_fake_static_test": int8_metrics,
        "training": {
            "model_name": opts.model_name,
            "block_size": args.block_size,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
        },
    }

    if trainer.is_world_process_zero():
        save_run_artifacts(trained_model, tokenizer, run_dir, opts, metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT-2 variants on WikiText-2 with optional mods; eval FP32 and INT8 PPL."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/train_params")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        choices=SUPPORTED_GPT2_MODELS,
        help="GPT-2 variant to train.",
    )

    parser.add_argument(
        "--optimizer",
        choices=["adamw", "soap"],
        default="adamw",
        help="Optimizer (SOAP needs pytorch-optimizer).",
    )
    parser.add_argument(
        "--attention-bias",
        action="store_true",
        help="Systematic-outliers K/V attention bias (https://github.com/an-yongqi/systematic-outliers).",
    )
    parser.add_argument(
        "--attention-linear-bias",
        action="store_true",
        help="Enable linear bias on GPT-2 attention Conv1D (disabled by default).",
    )
    parser.add_argument(
        "--context-aware-scaling",
        action="store_true",
        help="Context-aware scaling factor S(c) on attention output.",
    )
    parser.add_argument(
        "--op-blocks",
        action="store_true",
        help="Outlier-protected blocks from simplified_transformers (OP config).",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Quantization-aware training with int8 fake-quant (STE) during training.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy (0 = disabled).",
    )

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=64,
        help="Sliding-window stride for val/test perplexity (must be <= --block-size).",
    )
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size for per-epoch validation eval.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--single-gpu", action="store_true", help="Disable DDP even if multiple GPUs are visible.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    metrics = train_and_evaluate(args)
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run(parse_args())
