"""The training worker: runs inside each torchrun rank (or alone).

Reads the resolved config from the run directory (written by the `train`
command before launch), then:

    1. loads + profiles the dataset (token stats, coverage, preview)
    2. loads the prepared NF4 checkpoint onto its GPU
    3. freezes / casts / enables gradient checkpointing, attaches LoRA
    4. builds SFTConfig + SFTTrainer with the observability callbacks
    5. trains, saves the adapter, writes summary + charts + report

Rank discipline: rank 0 owns the terminal UI, metrics.jsonl, charts and the
report. Other ranks log warnings to their own file and stay quiet.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .config import Config
from .explain import kv_table
from .utils import fmt_int, human_duration, seed_everything, write_json

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from rich.console import Console
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from trl import SFTConfig, SFTTrainer

    from .data import DatasetStats, TokenProfile
    from .explain import ExplainerBook
    from .monitor import GpuSampler, MetricsCallback, MetricsWriter, RichDashboard, RunSummary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Worker CLI: --run-dir, --dry-run, --max-steps, --resume."""
    parser = argparse.ArgumentParser(description="rocm-qlora training worker")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.resolved.json")
    parser.add_argument("--dry-run", action="store_true", help="3 steps, tiny slice, no eval")
    parser.add_argument("--max-steps", type=int, default=None, help="Override optimizer steps")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory to resume from (optimizer/scheduler/step state included)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(run_dir: Path, rank: int) -> logging.Logger:
    """Rank log file + level (ROCM_QLORA_LOG_LEVEL); silences the known torch FutureWarning."""
    log_file = run_dir / "logs" / f"rank-{rank}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rocm_qlora")
    # Health-analysis verbosity: INFO by default; ROCM_QLORA_LOG_LEVEL=DEBUG
    # (or WARNING) adjusts the file logs without touching the dashboard.
    level_name = os.environ.get("ROCM_QLORA_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    # torch 2.13 renamed `all_gather_into_tensor` -> `all_gather_single`; HF
    # Trainer / Accelerate still call the old name once per logging step,
    # flooding the console with one FutureWarning per rank per step (~84
    # lines/run) and burying real signals. Silence precisely that message;
    # remove when the stack updates.
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*all_gather_into_tensor.*is deprecated.*")
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    if rank != 0:
        # keep stdout/stderr quiet on non-zero ranks; warnings still go to the file
        logging.getLogger("transformers").setLevel(logging.WARNING)
    return logger


# ---------------------------------------------------------------------------
# SFT config
# ---------------------------------------------------------------------------


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


def _skip_redundant_init_sync(log: Callable[[str], None]) -> None:
    """Disable DDP's initial parameter broadcast (safe *by construction* here).

    DDP broadcasts all module states from rank 0 at construction so replicas
    start identical. That broadcast coalesces the 16-bit parameters into one
    transient buffer — measured at ~4.9 GiB for this model (the fp16
    embed_tokens + lm_head pair), which does not fit on a 24 GB card next to
    the ~19 GiB of loaded weights (it OOMs before the first step).

    The broadcast is redundant in this project: every rank loads the *same*
    prepared checkpoint from disk, takes the *same* code path, and (with a
    shared seed) initializes LoRA identically — the replicas are bit-identical
    by construction. `torch.nn.parallel.distributed` is patched at its
    module-level binding; if a future torch renames it, this silently no-ops
    and the original broadcast (and its loud OOM) returns.
    """
    import torch.nn.parallel.distributed as ddp_mod

    if hasattr(ddp_mod, "_sync_module_states"):
        ddp_mod._sync_module_states = lambda *args, **kwargs: None  # pyright: ignore[reportPrivateUsage]  # deliberate no-op
        log("DDP init broadcast disabled (identical checkpoints + shared seed → replicas identical by construction)")
    else:
        log(
            "WARNING: torch.nn.parallel.distributed._sync_module_states is gone — the DDP init broadcast "
            "(~4.9 GiB transient at 27B) is back; expect an OOM at trainer construction"
        )


def _install_grad_finite_checks(model: torch.nn.Module, log: Callable[[str], None]) -> None:
    """Abort at the *exact* parameter whose gradient goes non-finite.

    The failure this exists for: a step with a finite loss can still produce
    NaN gradients in the backward. DDP allreduces the NaN into every rank,
    the optimizer step poisons the weights, and from then on the fused loss
    reports exactly 0.0 while the loss EMA keeps falling. Catching the NaN in
    the post-accumulate hook fires *before* the optimizer step and names the
    tensor — the difference between "grad_norm became NaN" and a root cause.

    Diagnostics only (env-gated): one isfinite kernel + sync per trainable
    tensor per micro-batch.
    """
    import torch

    def make_hook(param_name: str):
        def hook(grad: torch.Tensor):
            if grad is not None and not bool(torch.isfinite(grad).all()):  # pyright: ignore[reportUnnecessaryComparison]  # defensive
                raise RuntimeError(
                    f"non-finite gradient in '{param_name}' — aborting before the optimizer step "
                    "(ROCM_QLORA_GRAD_CHECK=1). See docs/05-troubleshooting.md"
                )

        return hook

    n = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_post_accumulate_grad_hook(make_hook(name))
            n += 1
    log(f"gradient finiteness checks installed on {n} trainable tensors")


def _sft_warmup_steps(cfg: Config, n_train: int, world_size: int, max_steps: int | None) -> int:
    """Warmup steps from the estimated total optimizer steps.

    transformers v5 removed `warmup_ratio` — the ratio is computed here.
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
        # `offload` / `every_n_layers` ride inside the kwargs dict: the trainer
        # pops them and forwards them to `gradient_checkpointing_enable`
        # (transformers >= 5.0; see its docstring for the trade-offs).
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
        # Full checkpoints (adapter + optimizer/scheduler/rng state, ~0.5 GB
        # each at 27B) so an aborted run can truly resume with
        # `rocm-qlora train --resume <run-dir>`. Epoch cadence by default;
        # `logging.checkpoint_every_n_steps > 0` switches to step cadence — on
        # flaky hardware (GPU resets) the epoch-only default can lose an hour.
        "save_strategy": "no" if dry_run else ("steps" if cfg.logging.checkpoint_every_n_steps else "epoch"),
        "save_only_model": False,
        "save_total_limit": None if cfg.logging.keep_epoch_checkpoints else 1,
        "eval_strategy": "epoch" if eval_enabled else "no",
        "report_to": [],
        "disable_tqdm": True,
        # HF/TRL internals verbosity: WARNING by default; set
        # ROCM_QLORA_HF_LOG_LEVEL=info to surface per-step internals in the
        # rank logs for health analysis.
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
        from dataclasses import fields

        field = next(f for f in fields(sft_config_cls) if f.name == "include_num_input_tokens_seen")
        kwargs["include_num_input_tokens_seen"] = "non_padding" if isinstance(field.default, str) else True
    except Exception:
        pass


def build_sft_config(
    cfg: Config, run_dir: Path, *, world_size: int, n_train: int, dry_run: bool, max_steps: int | None
):
    """SFTConfig for this run (see _sft_kwargs for the section mapping)."""
    from trl import SFTConfig

    warmup_steps = _sft_warmup_steps(cfg, n_train, world_size, max_steps)
    kwargs = _sft_kwargs(
        cfg, run_dir, world_size=world_size, dry_run=dry_run, max_steps=max_steps, warmup_steps=warmup_steps
    )
    _set_tokens_seen_kwarg(kwargs, SFTConfig)
    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# Rank-0 terminal stack
# ---------------------------------------------------------------------------


def _rank0_ui(cfg: Config, run_dir: Path, world_size: int, *, enabled: bool) -> _Rank0UI | None:
    """Console, explainer book, GPU sampler, metrics writer, live dashboard.

    Rank 0 only; non-zero ranks get `None` and log to their own rank file
    instead (`setup_logging`).
    """
    if not enabled:
        return None
    from rich.console import Console

    from .explain import ExplainerBook
    from .monitor import GpuSampler, MetricsWriter, RichDashboard

    console = Console()
    book = ExplainerBook(console, enabled=cfg.run.explainers)
    sampler = GpuSampler(cfg.logging.gpu_sample_seconds)
    writer = MetricsWriter(run_dir / "metrics.jsonl")
    dashboard = RichDashboard(console, refresh_per_second=cfg.run.console_refresh_per_second)
    dashboard.state.run_name = run_dir.name
    dashboard.state.model = cfg.model.name.split("/")[-1]
    dashboard.state.world_size = world_size
    dashboard.update(phase="loading data")
    return _Rank0UI(console, book, sampler, writer, dashboard)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


class _Rank0UI(NamedTuple):
    """The rank-0 terminal stack (``None`` on every other rank)."""

    console: Console
    book: ExplainerBook
    sampler: GpuSampler
    writer: MetricsWriter
    dashboard: RichDashboard


class _TrainingData(NamedTuple):
    """Phase-1 outputs consumed by the later training phases."""

    tokenizer: PreTrainedTokenizerBase
    original_chat_template: str | dict[str, Any] | None
    train_ds: Dataset
    eval_ds: Dataset | None
    n_train: int
    total_tokens_estimate: int


def _load_training_data(
    cfg: Config, args: argparse.Namespace, *, ui: _Rank0UI | None, logger: logging.Logger
) -> _TrainingData:
    """Phase 1: tokenizer + chat template, dataset load/split/profile, rank-0 summary."""
    from .data import load_sharegpt, profile_tokens, split_dataset, to_hf_dataset
    from .model import load_tokenizer

    tokenizer = load_tokenizer(cfg)
    original_chat_template = tokenizer.chat_template
    ensure_training_chat_template(tokenizer, cfg, logger.info)
    examples, stats = load_sharegpt(cfg.data.path, cfg.data.messages_field)
    profile = profile_tokens(
        tokenizer, examples, messages_field=cfg.data.messages_field, max_length=cfg.data.max_length
    )

    ds = to_hf_dataset(examples, cfg.data.messages_field)
    split = split_dataset(ds, eval_fraction=cfg.data.eval_fraction, seed=cfg.run.seed)
    train_ds = split["train"]
    eval_ds = split.get("test")
    if args.dry_run:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        eval_ds = None

    n_train = len(train_ds)
    total_tokens_estimate = int(profile.total_tokens * (n_train / max(1, profile.n_total)))

    if ui is not None:  # ui exists exactly on rank 0
        _show_dataset_panel(ui, cfg, stats, profile, n_train, eval_ds, examples, tokenizer, logger)

    return _TrainingData(tokenizer, original_chat_template, train_ds, eval_ds, n_train, total_tokens_estimate)


def _show_dataset_panel(
    ui: _Rank0UI,
    cfg: Config,
    stats: DatasetStats,
    profile: TokenProfile,
    n_train: int,
    eval_ds: Dataset | None,
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> None:
    """Rank-0 dataset summary panel, explainer cards and first-example preview."""
    from rich.panel import Panel

    from .data import render_preview

    console, book = ui.console, ui.book
    pcts = profile.percentiles()
    table = kv_table()
    table.add_row(
        "examples", f"{stats.n_examples} (train {n_train}, eval {len(eval_ds) if eval_ds is not None else 0})"
    )
    table.add_row("turns", ", ".join(f"{k}: {v}" for k, v in stats.turns_per_example.items()))
    table.add_row(
        "tokens/example",
        " ".join(f"p{k}={pcts[f'p{k}']:.0f}" for k in (50, 90, 95, 99)) + f" max={pcts['p100']:.0f}",
    )
    table.add_row(
        f"coverage @ max_length={cfg.data.max_length}",
        f"{profile.coverage_at(cfg.data.max_length) * 100:.1f}% "
        f"({profile.n_truncated} truncated, {profile.truncation_pct:.1f}%)",
    )
    table.add_row(
        "reasoning traces",
        f"{stats.think_blocks} assistant turns with  thinking ({stats.think_complete} complete)",
    )
    console.print(Panel(table, title="[bold]dataset[/bold]", border_style="cyan"))
    book.show("reasoning_data")
    book.show("truncation")
    if stats.n_examples < 1000:
        book.show("epochs_small_data")
    console.print(
        Panel(
            render_preview(tokenizer, examples[0], messages_field=cfg.data.messages_field),
            title="first example as the model sees it",
            border_style="blue",
        )
    )
    logger.info("dataset ready: %d train / %s eval", n_train, len(eval_ds) if eval_ds is not None else 0)


def _build_model(
    cfg: Config,
    local_rank: int,
    *,
    ui: _Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> PreTrainedModel | PeftModel:
    """Phase 2: attention verdict, QLoRA preparation, rank-0 model panel."""
    import torch

    if ui is not None:  # ui exists exactly on rank 0
        ui.dashboard.update(phase="loading model")
        ui.book.show("prepared_checkpoint")
        ui.book.show("hybrid_attention")
        ui.book.show("rocm_attention")

    from .launcher import apply_sdpa_verdict, resolve_attn_impl

    attn = resolve_attn_impl(cfg)
    disabled = apply_sdpa_verdict()
    if disabled:
        logger.info(
            "SDPA backends disabled by the doctor verdict: %s — masked attention falls back to math (docs/05)",
            ", ".join(disabled),
        )
    logger.info(
        "SDPA backends: flash=%s mem_efficient=%s math=%s",
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    from .model import attach_lora, load_model_for_training, parameter_report, prepare_for_qlora

    model = load_model_for_training(cfg, local_rank=local_rank, attn=attn)
    model = prepare_for_qlora(model, cfg)
    model = attach_lora(model, cfg)
    report = parameter_report(cast("PreTrainedModel", model))  # pyright: ignore[reportInvalidCast]  # PeftModel forwards to the base

    # --- optional numerical diagnostics (env-gated) ----------------------
    # Both exist because of a real failure mode: a run can hit a step with
    # a finite loss but NaN gradients; DDP then spreads the NaN to every
    # rank, the optimizer poisons the weights, and from then on the fused
    # loss reports exactly 0.0 (zero on ignored positions) while the EMA
    # keeps "improving". The on_log guard aborts, but *these* pinpoint.
    if os.environ.get("ROCM_QLORA_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
        logger.info("autograd anomaly detection ON — the first NaN-producing op will raise with a traceback")
    if os.environ.get("ROCM_QLORA_GRAD_CHECK") == "1":
        _install_grad_finite_checks(model, logger.info)

    # Reclaim the load-time casting transients still sitting in the
    # allocator's reserved pool (~2.4 GB for this model). Without this,
    # 27B training on 24 GB cards starts with almost no headroom.
    torch.cuda.empty_cache()

    if ui is not None:  # ui exists exactly on rank 0
        _show_model_panel(ui, attn, disabled, report, world_size=world_size)
    logger.info("model ready: %d trainable params", report["trainable"])
    return model


def _show_model_panel(ui: _Rank0UI, attn: str, disabled: list[str], report: dict[str, Any], *, world_size: int) -> None:
    """Rank-0 model summary panel, explainer cards and model_loaded event."""
    from rich.panel import Panel

    console, book = ui.console, ui.book
    table = kv_table()
    table.add_row("total params", fmt_int(report["total"]))
    table.add_row("trainable (LoRA)", f"{fmt_int(report['trainable'])}  ({report['pct']:.3f}%)")
    breakdown = " · ".join(f"{k}: {fmt_int(v)}" for k, v in list(report["by_suffix"].items())[:8])
    table.add_row("by module", breakdown)
    attn_note = f"{attn}  (doctor-verified where available)"
    if disabled:
        attn_note = f"{attn}  ({', '.join(disabled)} off per doctor verdict)"
    table.add_row("attention", attn_note)
    console.print(Panel(table, title="[bold]model[/bold]", border_style="cyan"))
    book.show("lora_targets")
    book.show("rank_alpha")
    book.show("assistant_only_loss")
    book.show("chunked_loss")
    book.show("multi_gpu_ddp")
    if ui.writer:
        from .monitor import EventRecord

        ui.writer.write(
            EventRecord(
                type="event",
                level="info",
                event="model_loaded",
                total_params=report["total"],
                trainable=report["trainable"],
                attn=attn,
                sdpa_disabled=disabled,
                world_size=world_size,
            )
        )


def _render_charts(*args: Any, **kwargs: Any) -> None:
    """Lazy chart renderer injected into MetricsCallback (keeps `plots` out of `monitor`)."""
    from .plots import render_charts

    render_charts(*args, **kwargs)


def _build_trainer(
    cfg: Config,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    data: _TrainingData,
    model: PreTrainedModel | PeftModel,
    ui: _Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> tuple[SFTTrainer, MetricsCallback | None]:
    """Phase 3: SFTConfig, observability callback, init-sync skip, SFTTrainer."""
    if ui is not None:  # ui exists exactly on rank 0
        ui.dashboard.update(phase="building trainer")
        ui.book.show("effective_batch")
        ui.book.show("warmup")
        ui.book.show("vram_budget")

    sft_config = build_sft_config(
        cfg, run_dir, world_size=world_size, n_train=data.n_train, dry_run=args.dry_run, max_steps=args.max_steps
    )
    if ui is not None:  # ui exists exactly on rank 0
        eff = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
        ui.console.print(
            f"[dim]effective batch: {cfg.optim.per_device_batch_size} (per GPU) × "
            f"{cfg.optim.grad_accum} (accum) × {world_size} (GPUs) = [bold]{eff}[/bold] sequences/step[/dim]"
        )

    callbacks = []
    metrics_callback = None
    if ui is not None:  # ui exists exactly on rank 0
        from .monitor import MetricsCallback

        metrics_callback = MetricsCallback(
            cfg,
            run_dir,
            console=ui.console,
            dashboard=ui.dashboard,
            book=ui.book,
            sampler=ui.sampler,
            writer=ui.writer,
            total_tokens_estimate=data.total_tokens_estimate,
            world_size=world_size,
            chart_renderer=_render_charts,
        )
        callbacks = [metrics_callback]

    if world_size > 1:
        _skip_redundant_init_sync(logger.info)

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=data.train_ds,
        eval_dataset=data.eval_ds,
        processing_class=data.tokenizer,
        callbacks=cast("list[Any]", callbacks),  # duck-typed hooks; not a TrainerCallback subclass
    )
    return trainer, metrics_callback


def _train_and_save(
    cfg: Config,
    run_dir: Path,
    args: argparse.Namespace,
    trainer: SFTTrainer,
    *,
    data: _TrainingData,
    ui: _Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> Path:
    """Phase 4: dashboard start, train (resumable), save adapter + inference template."""
    import torch

    if ui is not None:  # ui exists exactly on rank 0
        ui.dashboard.update(phase="training")
        ui.dashboard.start()
        ui.book.show("reading_loss")
        ui.book.show("grad_norm")
        ui.book.show("gpu_health")
        if data.eval_ds is not None:
            ui.book.show("eval")
        ui.book.hint(f"charts refresh every {cfg.logging.chart_snapshot_steps} steps into {run_dir / 'charts'}")

    if args.resume:
        logger.info("resuming from checkpoint: %s", args.resume)
    trainer.train(resume_from_checkpoint=args.resume)

    if ui is not None:  # ui exists exactly on rank 0
        ui.dashboard.update(phase="saving adapter", step=trainer.state.global_step)
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    # Save the *inference* template with the adapter (the training template
    # with generation markers is an implementation detail of the loss mask).
    data.tokenizer.chat_template = data.original_chat_template
    data.tokenizer.save_pretrained(adapter_dir)
    logger.info("adapter saved to %s", adapter_dir)

    if world_size > 1:
        with contextlib.suppress(Exception):
            torch.distributed.barrier()
    return adapter_dir


def _report_failure(
    run_dir: Path, dashboard: RichDashboard | None, writer: MetricsWriter | None, error: Exception
) -> None:
    """Rank-0 abort path: stop the dashboard, record the abort event, print the panel."""
    if dashboard:
        with contextlib.suppress(Exception):
            dashboard.stop()
    if writer:
        from .monitor import EventRecord

        writer.write(EventRecord(type="event", level="error", event="abort", error=str(error)))
        writer.close()
    from rich.console import Console
    from rich.panel import Panel

    Console().print(
        Panel(
            f"[bold red]Run aborted:[/bold red] {error}\n\n"
            f"Full logs: {run_dir / 'logs'}\n"
            "Troubleshooting guide: docs/05-troubleshooting.md",
            title="[red]failure[/red]",
            border_style="red",
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Worker entry: data → model → trainer → train → finalize (rank 0 owns the UI)."""
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    cfg = Config.from_dict(json.loads((run_dir / "config.resolved.json").read_text()))

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    logger = setup_logging(run_dir, rank)
    # Identical seed on every rank: required so all replicas initialize
    # identically (see _skip_redundant_init_sync).
    seed_everything(cfg.run.seed)

    import torch

    torch.cuda.set_device(local_rank)

    start_time = time.monotonic()
    ui = _rank0_ui(cfg, run_dir, world_size, enabled=is_main)

    try:
        data = _load_training_data(cfg, args, ui=ui, logger=logger)
        model = _build_model(cfg, local_rank, ui=ui, world_size=world_size, logger=logger)
        trainer, metrics_callback = _build_trainer(
            cfg, run_dir, args, data=data, model=model, ui=ui, world_size=world_size, logger=logger
        )
        adapter_dir = _train_and_save(
            cfg, run_dir, args, trainer, data=data, ui=ui, world_size=world_size, logger=logger
        )

        if ui is not None:  # ui exists exactly on rank 0
            _finalize_run(
                cfg,
                run_dir,
                adapter_dir,
                ui,
                metrics_callback,
                start_time=start_time,
                world_size=world_size,
                dry_run=args.dry_run,
            )

    except Exception as exc:
        logger.error("training failed: %s\n%s", exc, traceback.format_exc())
        if ui is not None:  # ui exists exactly on rank 0
            _report_failure(run_dir, ui.dashboard, ui.writer, exc)
        raise
    finally:
        if world_size > 1:
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Run finalization (rank 0)
# ---------------------------------------------------------------------------


def _finalize_run(
    cfg: Config,
    run_dir: Path,
    adapter_dir: Path,
    ui: _Rank0UI,
    metrics_callback: MetricsCallback | None,
    *,
    start_time: float,
    world_size: int,
    dry_run: bool,
) -> None:
    """Rank-0 end of run: summary.json, run_end event, charts, report, wrap-up."""
    from .monitor import summarize
    from .plots import render_charts, write_report

    ui.dashboard.update(phase="finalizing")
    ui.dashboard.stop()

    records = metrics_callback.records if metrics_callback else []
    summary = summarize(records)
    summary.total_seconds = time.monotonic() - start_time
    summary.run_dir = str(run_dir)
    summary.adapter_dir = str(adapter_dir)
    summary.model = cfg.model.name
    summary.dataset = cfg.data.path
    summary.effective_batch = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
    summary.world_size = world_size
    summary.max_length = cfg.data.max_length
    summary.lora_r = cfg.lora.r
    summary.lr = cfg.optim.lr
    summary.epochs = cfg.optim.epochs
    summary.loss_type = cfg.optim.loss_type
    summary.dry_run = dry_run
    write_json(run_dir / "summary.json", summary.to_dict())
    if ui.writer:
        from .monitor import RunEndRecord

        run_end: RunEndRecord = {
            "type": "run_end",
            "total_seconds": summary.total_seconds,
            "steps": summary.steps,
            "best_loss": summary.best_loss,
            "best_eval_loss": summary.best_eval_loss,
            "tokens_seen": summary.tokens_seen,
        }
        ui.writer.write(run_end)
        ui.writer.close()

    render_charts(
        run_dir,
        records=records,
        meta={
            "run_name": run_dir.name,
            "model": cfg.model.name.split("/")[-1],
            "max_grad_norm": cfg.optim.max_grad_norm,
        },
    )
    write_report(run_dir)

    _print_wrapup(ui.console, ui.book, run_dir, summary, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------


def _short_path(path: Path) -> str:
    """cwd-relative when possible — keeps wrap-up lines short and copy-pasteable."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _print_wrapup(console: Console, book: ExplainerBook, run_dir: Path, summary: RunSummary, *, dry_run: bool) -> None:
    """Run-complete panel: headline numbers + next-step hints."""
    from rich.panel import Panel

    table = kv_table()
    table.add_row("steps", fmt_int(summary.steps))
    nan = float("nan")
    table.add_row(
        "loss",
        f"{(summary.first_loss if summary.first_loss is not None else nan):.4f} → "
        f"{(summary.last_loss if summary.last_loss is not None else nan):.4f} "
        f"(best {(summary.best_loss if summary.best_loss is not None else nan):.4f} @ step {summary.best_step})",
    )
    if summary.last_eval_loss is not None:
        table.add_row(
            "eval loss",
            f"{(summary.first_eval_loss if summary.first_eval_loss is not None else nan):.4f} → "
            f"{summary.last_eval_loss:.4f} "
            f"(best {(summary.best_eval_loss if summary.best_eval_loss is not None else nan):.4f})",
        )
    table.add_row("tokens seen", fmt_int(summary.tokens_seen))
    table.add_row("wall time", human_duration(summary.total_seconds))
    table.add_row(
        "peak VRAM (max GPU)", f"{(summary.peak_vram_gb if summary.peak_vram_gb is not None else nan):.1f} GB"
    )
    table.add_row("adapter", _short_path(Path(summary.adapter_dir)))
    table.add_row("charts", _short_path(run_dir / "charts"))
    table.add_row("report", _short_path(run_dir / "report.md"))
    console.print(Panel(table, title="[bold green]run complete[/bold green]", border_style="green"))
    if dry_run:
        console.print("[yellow]This was a dry run (3 steps). Launch without --dry-run for the real thing.[/yellow]")
    book.show("next_steps")
    short = _short_path(run_dir)
    console.print("[dim]next:[/dim]")
    console.print(f"  [bold]rocm-qlora chat   --run {short}[/bold]")
    console.print(f"  [bold]rocm-qlora merge  --run {short}[/bold]")
    console.print(f"  [bold]rocm-qlora report --run {short}[/bold]")


if __name__ == "__main__":
    main()
