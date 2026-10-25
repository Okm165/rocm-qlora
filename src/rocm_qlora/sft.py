"""SFTConfig assembly: project config -> TRL trainer arguments.

Everything here maps ``Config`` fields onto ``trl.SFTConfig``. The training
worker (``train.py``) imports this; nothing here touches the GPU or the UI.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _bundled_training_template(name: str) -> str | None:
    """TRL's bundled training template for a config-named family, or None.

    The name is `data.training_template` — the user's declaration, not a guess
    from the checkpoint's id or its template's contents. None means the family
    is unknown (the config's Literal already restricts the values, so this is
    a TRL rename) and the caller raises.
    """
    from trl.chat_template_utils import (
        qwen3_5_nothink_training_chat_template,
        qwen3_5_think_training_chat_template,
        qwen3_8_training_chat_template,
        qwen3_training_chat_template,
    )

    return {
        "qwen3.8": qwen3_8_training_chat_template,
        "qwen3.5_think": qwen3_5_think_training_chat_template,
        "qwen3.5_nothink": qwen3_5_nothink_training_chat_template,
        "qwen3": qwen3_training_chat_template,
    }.get(name)


def ensure_training_chat_template(tokenizer: PreTrainedTokenizerBase, cfg: Config, log: Callable[[str], None]) -> None:
    """Make TRL's `{% generation %}` template detection succeed.

    TRL's SFTTrainer patches a template only when it lacks generation markers,
    and its patch helper matches the checkpoint's copy against bundled
    references by exact string equality. Asking the same helper covers the
    stock case; community copies (e.g. unsloth's Qwen3.8-27B) fail that match
    and get the template `data.training_template` names instead. With no
    family configured the run aborts — it never guesses a template, because
    the template decides which tokens the loss is computed on.
    """
    if not cfg.data.assistant_only_loss:
        return

    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    if has_generation_markers(tokenizer.chat_template):
        return  # TRL's own gate: a marker-bearing template is used as-is
    # Resolve the configured family *before* the try: a TRL rename of a bundled
    # constant must surface as itself, not masked inside an except handler.
    fallback = _bundled_training_template(cfg.data.training_template)
    try:
        get_training_chat_template(tokenizer)
    except ValueError as exc:
        # Only the fallback needs the tokenizer's template replaced (TRL's
        # trainer patches its own copy in the stock case, leaving the saved
        # processing class — and every mid-run checkpoint — with the original).
        if fallback is None:
            raise RuntimeError(
                f"TRL cannot patch {cfg.model.name}'s chat template — set data.training_template to this "
                "checkpoint's family (docs/03 explains how to tell) or set data.assistant_only_loss = false"
            ) from exc
        tokenizer.chat_template = fallback
        log(f"installed TRL's bundled {cfg.data.training_template} training chat template")


def _sft_kwargs(cfg: Config, run_dir: Path, *, dry_run: bool, max_steps: int | None) -> dict[str, Any]:
    """The SFTConfig kwargs: a declarative mapping from our config sections."""
    eval_enabled = cfg.eval.per_epoch and cfg.data.eval_fraction > 0 and not dry_run
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "adapter"),
        # A dry run is a smoke test: short sequences keep the three steps quick.
        "max_length": min(cfg.data.max_length, 512) if dry_run else cfg.data.max_length,
        "truncation_mode": cfg.data.truncation_mode,
        "packing": cfg.data.packing,
        "packing_strategy": cfg.data.packing_strategy,
        "assistant_only_loss": cfg.data.assistant_only_loss,
        "loss_type": cfg.optim.loss_type,
        "dataset_num_proc": cfg.data.dataset_num_proc,
        "eval_packing": cfg.eval.eval_packing,
        "per_device_train_batch_size": cfg.optim.per_device_batch_size,
        "per_device_eval_batch_size": cfg.eval.per_device_batch_size,
        "gradient_accumulation_steps": cfg.optim.grad_accum,
        "gradient_checkpointing": cfg.optim.gradient_checkpointing,
        # The trainer pops these and forwards them to gradient_checkpointing_enable.
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
            "offload": cfg.optim.activation_offloading,
            "every_n_layers": cfg.optim.checkpoint_every_n_layers,
        },
        "learning_rate": cfg.optim.lr,
        "lr_scheduler_type": cfg.optim.scheduler,
        # transformers v5 reads a warmup_steps float in [0, 1) as the ratio of
        # total optimizer steps (its own ceil arithmetic), so no schedule math
        # lives here.
        "warmup_steps": cfg.optim.warmup_ratio,
        "max_grad_norm": cfg.optim.max_grad_norm,
        "weight_decay": cfg.optim.weight_decay,
        "num_train_epochs": cfg.optim.epochs,
        "optim": cfg.optim.optimizer,
        "bf16": cfg.model.dtype == "bfloat16",
        "fp16": cfg.model.dtype == "float16",
        "logging_steps": 1,
        "logging_first_step": True,
        # Full checkpoints (adapter + optimizer/scheduler/rng state) so an
        # aborted run can truly resume; epoch cadence by default,
        # `logging.checkpoint_every_n_steps > 0` switches to step cadence.
        "save_strategy": "no" if dry_run else ("steps" if cfg.logging.checkpoint_every_n_steps else "epoch"),
        "save_only_model": False,
        "save_total_limit": None if cfg.logging.keep_epoch_checkpoints else 1,
        "eval_strategy": "epoch" if eval_enabled else "no",
        "report_to": [],
        "disable_tqdm": True,
        "log_level": os.environ.get("ROCM_QLORA_HF_LOG_LEVEL", "warning"),
        "seed": cfg.run.seed,
        "data_seed": cfg.run.seed,
        # The dataset is tiny; two loader processes per rank are plenty and
        # do not fight the training ranks for host cores.
        "dataloader_num_workers": 2,
        "ddp_find_unused_parameters": False,
        "torch_empty_cache_steps": 50,
        # transformers v5 accepts "all" | "non_padding" | "no"; the callback
        # reads the resulting num_input_tokens_seen from the log.
        "include_num_input_tokens_seen": "non_padding",
    }
    if cfg.logging.checkpoint_every_n_steps and not dry_run:
        kwargs["save_steps"] = cfg.logging.checkpoint_every_n_steps
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if dry_run:
        kwargs["max_steps"] = max_steps or 3
    return kwargs


def build_sft_config(cfg: Config, run_dir: Path, *, dry_run: bool, max_steps: int | None):
    """SFTConfig for this run (see _sft_kwargs for the section mapping)."""
    from trl import SFTConfig

    return SFTConfig(**_sft_kwargs(cfg, run_dir, dry_run=dry_run, max_steps=max_steps))
