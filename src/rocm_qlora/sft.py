"""SFTConfig assembly: project config -> TRL trainer arguments.

Everything here maps ``Config`` fields onto ``trl.SFTConfig`` and computes the
warmup steps transformers v5 no longer derives from a ratio. The training
worker (``train.py``) imports this; nothing here touches the GPU or the UI.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from trl import SFTConfig


def ensure_training_chat_template(
    tokenizer: PreTrainedTokenizerBase, cfg: Config, log: Callable[[str], None]
) -> str | None:
    """Install a `{% generation %}`-marked training template when TRL's
    auto-detection cannot match the checkpoint's template.

    TRL swaps in a training template for known model families by *exact
    string equality* against its bundled reference templates. Community
    checkpoints (e.g. unsloth's Qwen3.8-27B) often carry a slightly modified
    template, so the match fails and TRL refuses `assistant_only_loss=True`.
    The bundled family template is then picked by model name — the same file
    TRL would have swapped in.
    """
    import trl

    if not cfg.data.assistant_only_loss:
        return None
    template = getattr(tokenizer, "chat_template", None) or ""
    if "{% generation %}" in template:
        return None  # already training-compatible

    name = cfg.model.name.lower()
    candidates: list[str] = []
    if "qwen3.8" in name or "qwen3_8" in name:
        candidates.append("qwen3_8_training.jinja")
    if "qwen3.5" in name or "qwen3_5" in name:
        candidates.append("qwen3_5_think_training.jinja")
    if "qwen3" in name:
        candidates.append("qwen3_training.jinja")

    for fname in candidates:
        path = Path(trl.__file__).parent / "chat_templates" / fname
        if path.exists():
            tokenizer.chat_template = path.read_text(encoding="utf-8")
            log(f"installed training chat template: {fname} (auto-detection could not match the checkpoint template)")
            return fname

    log("WARNING: no bundled training template matched; disabling assistant_only_loss")
    cfg.data.assistant_only_loss = False
    return None


def _warmup_steps(cfg: Config, n_train: int, world_size: int, max_steps: int | None) -> int:
    """Warmup steps from the computed total optimizer steps (transformers v5
    removed `warmup_ratio`, so the ratio is applied here).

    Every term is known — dataset size, batch, accumulation, world size,
    epochs (or the explicit ``max_steps``) — so this is arithmetic, not an
    estimate.
    """
    effective_batch = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
    steps_per_epoch = max(1, -(-n_train // effective_batch))  # ceil division
    total_steps = max_steps or int(steps_per_epoch * cfg.optim.epochs)
    return max(1, int(total_steps * cfg.optim.warmup_ratio)) if cfg.optim.warmup_ratio else 0


def _sft_kwargs(
    cfg: Config, run_dir: Path, *, world_size: int, dry_run: bool, max_steps: int | None, warmup_steps: int
) -> dict[str, Any]:
    """The SFTConfig kwargs: a declarative mapping from our config sections."""
    eval_enabled = cfg.eval.per_epoch and cfg.data.eval_fraction > 0 and not dry_run
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "adapter"),
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
        "warmup_steps": warmup_steps,
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
        "dataloader_num_workers": min(2, (os.cpu_count() or 8) // max(1, world_size)),
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "torch_empty_cache_steps": 50,
    }
    if cfg.logging.checkpoint_every_n_steps and not dry_run:
        kwargs["save_steps"] = cfg.logging.checkpoint_every_n_steps
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if dry_run:
        kwargs["max_steps"] = max_steps or 3
    return kwargs


def _set_tokens_seen_kwarg(kwargs: dict[str, Any], sft_config_cls: type[SFTConfig]) -> None:
    """`include_num_input_tokens_seen` changed type across transformers versions
    (bool -> str enum with values all|non_padding|no)."""
    try:
        field = next(f for f in fields(sft_config_cls) if f.name == "include_num_input_tokens_seen")
        kwargs["include_num_input_tokens_seen"] = "non_padding" if isinstance(field.default, str) else True
    except Exception:
        pass


def build_sft_config(
    cfg: Config, run_dir: Path, *, world_size: int, n_train: int, dry_run: bool, max_steps: int | None
):
    """SFTConfig for this run (see _sft_kwargs for the section mapping)."""
    from trl import SFTConfig

    warmup = _warmup_steps(cfg, n_train, world_size, max_steps)
    kwargs = _sft_kwargs(cfg, run_dir, world_size=world_size, dry_run=dry_run, max_steps=max_steps, warmup_steps=warmup)
    _set_tokens_seen_kwarg(kwargs, SFTConfig)
    return SFTConfig(**kwargs)
