from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
MODEL_ID = "gpt2"


@dataclass
class TrainOptions:
    optimizer: str
    attention_bias: bool
    attention_linear_bias: bool
    context_aware_scaling: bool
    op_blocks: bool
    qat: bool
    label_smoothing: float

    def run_name(self) -> str:
        parts = [
            "gpt2small",
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
        optimizer=args.optimizer,
        attention_bias=args.attention_bias,
        attention_linear_bias=args.attention_linear_bias,
        context_aware_scaling=args.context_aware_scaling,
        op_blocks=args.op_blocks,
        qat=args.qat,
        label_smoothing=args.label_smoothing,
    )


def build_model(opts: TrainOptions) -> GPT2LMHeadModel:
    config = GPT2Config.from_pretrained(MODEL_ID)
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


class TrainParamsTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_options: TrainOptions,
        cli_args: argparse.Namespace,
        **kwargs,
    ) -> None:
        self.train_options = train_options
        self.cli_args = cli_args
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.model, self.cli_args)
        return self.optimizer


@torch.no_grad()
def eval_wikitext2_perplexity(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    block_size: int,
    stride: int,
    split: str = "test",
) -> Dict[str, float]:
    ds = load_dataset("wikitext", WIKITEXT_CONFIG, split=split)
    text = "\n\n".join(t for t in ds["text"] if isinstance(t, str) and t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    model.eval()

    for begin in tqdm(range(0, seq_len, stride), desc=f"ppl-{split}", leave=False):
        end = min(begin + block_size, seq_len)
        trg_len = end - begin
        if trg_len < 2:
            continue
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-trg_len] = -100
        outputs = model(chunk, labels=labels)
        if outputs.loss is None or not math.isfinite(float(outputs.loss)):
            continue
        nlls.append(outputs.loss.detach() * trg_len)

    if not nlls:
        raise RuntimeError(f"No valid loss chunks for WikiText-2 {split} perplexity.")

    total_nll = torch.stack(nlls).sum().item()
    avg_nll = total_nll / seq_len
    perplexity = math.exp(avg_nll) if avg_nll < 80 else float("inf")
    return {"perplexity": perplexity, "avg_nll": avg_nll, "tokens": seq_len}


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

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_blocks = load_wikitext_blocks("train", tokenizer, args.block_size)
    if not train_blocks["input_ids"]:
        raise RuntimeError("WikiText-2 train split produced zero blocks; check block_size.")

    train_ds = BlockDataset(train_blocks["input_ids"], train_blocks["attention_mask"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        label_smoothing_factor=opts.label_smoothing,
        **({"ddp_find_unused_parameters": False} if use_ddp else {}),
    )

    trainer = TrainParamsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        train_options=opts,
        cli_args=args,
    )

    logger.info("Training run=%s on %d block(s), ddp=%s", run_name, len(train_ds), use_ddp)
    trainer.train()
    if opts.qat:
        disable_qat_training()

    trained_model = trainer.model
    if hasattr(trained_model, "module"):
        trained_model = trained_model.module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)

    fp32_metrics = eval_wikitext2_perplexity(
        trained_model,
        tokenizer,
        device,
        block_size=min(args.block_size, trained_model.config.n_positions),
        stride=args.eval_stride,
        split="test",
    )
    logger.info("FP32 test perplexity: %.4f", fp32_metrics["perplexity"])

    int8_model = apply_int8_fake_quant(trained_model).to(device)
    int8_metrics = eval_wikitext2_perplexity(
        int8_model,
        tokenizer,
        device,
        block_size=min(args.block_size, int8_model.config.n_positions),
        stride=args.eval_stride,
        split="test",
    )
    logger.info("INT8 fake-quant test perplexity: %.4f", int8_metrics["perplexity"])

    metrics = {
        "run_name": run_name,
        "options": {
            "optimizer": opts.optimizer,
            "attention_bias": opts.attention_bias,
            "attention_linear_bias": opts.attention_linear_bias,
            "context_aware_scaling": opts.context_aware_scaling,
            "op_blocks": opts.op_blocks,
            "qat": opts.qat,
            "label_smoothing": opts.label_smoothing,
        },
        "fp32_test": fp32_metrics,
        "int8_fake_static_test": int8_metrics,
        "training": {
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
        description="Train GPT-2 small on WikiText-2 with optional mods; eval FP32 and INT8 PPL."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/train_params")

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
    parser.add_argument("--eval-stride", type=int, default=64)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
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
