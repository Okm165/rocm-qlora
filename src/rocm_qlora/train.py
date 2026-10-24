"""The training worker: runs inside each torchrun rank (or alone).

Reads the resolved config from the run directory (written by the `train`
command before launch), then:

    1. loads + profiles the dataset (token stats, coverage, preview)
    2. loads the prepared NF4 checkpoint onto its GPU
    3. freezes / casts / enables gradient checkpointing, attaches LoRA
    4. builds SFTConfig + SFTTrainer with the observability callbacks
    5. trains, saves the adapter, then renders charts + the Markdown report
       from the metrics stream

Rank discipline: rank 0 owns the terminal UI, metrics.jsonl, charts and the
report. Other ranks log warnings to their own file and stay quiet.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .config import Config
from .sft import build_sft_config, ensure_training_chat_template
from .utils import fmt_float, fmt_int, human_duration, seed_everything

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from rich.console import Console
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from trl import SFTTrainer

    from .data import DatasetStats, TokenProfile
    from .model import ParameterReport
    from .monitor import GpuSampler, MetricsCallback, MetricsRecord, MetricsWriter, RunSummary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerArgs:
    """The worker's launch contract (what the parent ``train`` command sends)."""

    run_dir: Path
    dry_run: bool
    max_steps: int | None
    resume: str | None


def parse_args(argv: list[str] | None = None) -> WorkerArgs:
    """Worker CLI: --run-dir, --dry-run, --max-steps, --resume."""
    parser = argparse.ArgumentParser(description="rocm-qlora training worker")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.resolved.toml")
    parser.add_argument("--dry-run", action="store_true", help="3 steps, tiny slice, no eval")
    parser.add_argument("--max-steps", type=int, default=None, help="Override optimizer steps")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory to resume from (optimizer/scheduler/step state included)",
    )
    ns = parser.parse_args(argv)
    return WorkerArgs(run_dir=Path(ns.run_dir), dry_run=ns.dry_run, max_steps=ns.max_steps, resume=ns.resume)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(run_dir: Path, rank: int) -> logging.Logger:
    """Rank log file + level (ROCM_QLORA_LOG_LEVEL); silences the known torch FutureWarning."""
    log_file = run_dir / "logs" / f"rank-{rank}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rocm_qlora")
    # Health-analysis verbosity: INFO by default; ROCM_QLORA_LOG_LEVEL=DEBUG
    # (or WARNING) adjusts the file logs without touching the console.
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
# Runtime patches
# ---------------------------------------------------------------------------


def _skip_redundant_init_sync(log: Callable[[str], None]) -> None:
    """Disable DDP's initial parameter broadcast (safe *by construction* here).

    The broadcast coalesces the 16-bit parameters into one ~4.9 GiB transient
    buffer (fp16 embed_tokens + lm_head), which OOMs a 24 GB card next to the
    loaded weights. It is redundant: every rank loads the same prepared
    checkpoint and initializes LoRA with the same seed, so the replicas are
    bit-identical by construction. If a future torch renames the patched
    binding, this silently no-ops and the broadcast (and its OOM) returns.
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


# ---------------------------------------------------------------------------
# Rank-0 terminal output
# ---------------------------------------------------------------------------


class Rank0UI(NamedTuple):
    """The rank-0 stack (``None`` on every other rank)."""

    console: Console
    sampler: GpuSampler
    writer: MetricsWriter


def _rank0_ui(cfg: Config, run_dir: Path, *, enabled: bool) -> Rank0UI | None:
    """Console, GPU sampler, metrics writer (rank 0 only).

    Non-zero ranks get ``None`` and log to their own rank file instead
    (``setup_logging``).
    """
    if not enabled:
        return None
    from rich.console import Console

    from .monitor import GpuSampler, MetricsWriter

    console = Console()
    sampler = GpuSampler(cfg.logging.gpu_sample_seconds)
    writer = MetricsWriter(run_dir / "metrics.jsonl")
    return Rank0UI(console, sampler, writer)


def _short_path(path: Path) -> str:
    """cwd-relative when possible — keeps status lines short and copy-pasteable."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _print_dataset_status(
    console: Console, cfg: Config, stats: DatasetStats, profile: TokenProfile, n_train: int, n_eval: int
) -> None:
    """Dataset summary lines (rank 0); the rendered-example preview is `rocm-qlora data`'s job."""
    pcts = profile.percentiles()
    turns = ", ".join(f"{k}: {v}" for k, v in stats.turns_per_example.items())
    console.print(
        f"[dim]dataset:[/dim] {fmt_int(stats.n_examples)} examples (train {n_train}, eval {n_eval}) · turns {turns}"
    )
    console.print(
        f"[dim]tokens/example:[/dim] p50={pcts.p50:.0f} p90={pcts.p90:.0f} p95={pcts.p95:.0f} "
        f"p99={pcts.p99:.0f} max={pcts.p100:.0f} · coverage {profile.coverage_at(cfg.data.max_length) * 100:.1f}% "
        f"@ {cfg.data.max_length} ({profile.n_truncated} truncated, {profile.truncation_pct:.1f}%)"
    )
    console.print(
        f"[dim]reasoning traces:[/dim] {stats.think_blocks} assistant turns with  thinking "
        f"({stats.think_complete} complete)"
    )


def _print_model_status(
    ui: Rank0UI, attn: str, disabled: list[str], report: ParameterReport, *, world_size: int
) -> None:
    """Model summary lines + the model_loaded event (rank 0)."""
    from .monitor import EventRecord

    attn_note = f"{attn} (doctor-verified where available)"
    if disabled:
        attn_note = f"{attn} ({', '.join(disabled)} off per doctor verdict)"
    ui.console.print(
        f"[dim]model:[/dim] {fmt_int(report['total'])} params · trainable {fmt_int(report['trainable'])} "
        f"({report['pct']:.3f}%) · attention {attn_note}"
    )
    breakdown = " · ".join(f"{k}: {fmt_int(v)}" for k, v in list(report["by_module"].items())[:8])
    ui.console.print(f"[dim]lora modules:[/dim] {breakdown}")
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


def _print_wrapup(console: Console, run_dir: Path, summary: RunSummary, *, dry_run: bool) -> None:
    """Run-complete lines: headline numbers, artifacts, next commands."""
    nan = float("nan")
    loss = (
        f"{(summary.first_loss if summary.first_loss is not None else nan):.4f} → "
        f"{(summary.last_loss if summary.last_loss is not None else nan):.4f} "
        f"(best {(summary.best_loss if summary.best_loss is not None else nan):.4f} @ step {summary.best_step})"
    )
    console.print(
        f"[bold green]run complete[/bold green]: {fmt_int(summary.steps)} steps · loss {loss} · "
        f"{fmt_int(summary.tokens_seen)} tokens · {human_duration(summary.total_seconds)} · "
        f"peak VRAM {(summary.peak_vram_gb if summary.peak_vram_gb is not None else nan):.1f} GB"
    )
    if summary.last_eval_loss is not None:
        console.print(
            f"[dim]eval loss:[/dim] {fmt_float(summary.first_eval_loss)} → {fmt_float(summary.last_eval_loss)} "
            f"(best {fmt_float(summary.best_eval_loss)})"
        )
    console.print(
        f"[dim]artifacts:[/dim] adapter {_short_path(Path(summary.adapter_dir))} · "
        f"charts {_short_path(run_dir / 'charts')} · report {_short_path(run_dir / 'report.md')}"
    )
    if dry_run:
        console.print("[yellow]This was a dry run (3 steps). Launch without --dry-run for the real thing.[/yellow]")
    short = _short_path(run_dir)
    console.print(f"[dim]next:[/dim] rocm-qlora chat --run {short} · merge --run {short} · report --run {short}")


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


class _TrainingData(NamedTuple):
    """Phase-1 outputs consumed by the later training phases."""

    tokenizer: PreTrainedTokenizerBase
    original_chat_template: str | dict[str, Any] | None
    train_ds: Dataset
    eval_ds: Dataset | None
    n_train: int


def _load_training_data(cfg: Config, args: WorkerArgs, *, ui: Rank0UI | None, logger: logging.Logger) -> _TrainingData:
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

    if ui is not None:  # ui exists exactly on rank 0
        _print_dataset_status(ui.console, cfg, stats, profile, n_train, len(eval_ds) if eval_ds is not None else 0)
        logger.info("dataset ready: %d train / %s eval", n_train, len(eval_ds) if eval_ds is not None else 0)

    return _TrainingData(tokenizer, original_chat_template, train_ds, eval_ds, n_train)


def _resolve_attention(cfg: Config, logger: logging.Logger) -> tuple[str, list[str]]:
    """Attention implementation + the SDPA backends the doctor verdict disabled."""
    import torch

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
    return attn, disabled


def _install_diagnostics(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Env-gated numerical diagnostics (see ``health_violation`` for the failure mode)."""
    if os.environ.get("ROCM_QLORA_DETECT_ANOMALY") == "1":
        import torch

        torch.autograd.set_detect_anomaly(True)
        logger.info("autograd anomaly detection ON — the first NaN-producing op will raise with a traceback")
    if os.environ.get("ROCM_QLORA_GRAD_CHECK") == "1":
        _install_grad_finite_checks(model, logger.info)


def _build_model(
    cfg: Config,
    local_rank: int,
    *,
    ui: Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> PreTrainedModel | PeftModel:
    """Phase 2: attention verdict, QLoRA preparation, rank-0 model lines."""
    import torch

    if ui is not None:  # ui exists exactly on rank 0
        ui.console.print("[dim]loading model (4-bit)…[/dim]")

    attn, disabled = _resolve_attention(cfg, logger)
    from .model import attach_lora, load_model_for_training, parameter_report, prepare_for_qlora

    model = load_model_for_training(cfg, local_rank=local_rank, attn=attn)
    model = prepare_for_qlora(model, cfg)
    model = attach_lora(model, cfg)
    report = parameter_report(cast("PreTrainedModel", model))  # pyright: ignore[reportInvalidCast]  # PeftModel forwards to the base

    _install_diagnostics(model, logger)

    # Reclaim the load-time casting transients still sitting in the
    # allocator's reserved pool (~2.4 GB for this model). Without this,
    # 27B training on 24 GB cards starts with almost no headroom.
    torch.cuda.empty_cache()

    if ui is not None:  # ui exists exactly on rank 0
        _print_model_status(ui, attn, disabled, report, world_size=world_size)
    logger.info("model ready: %d trainable params", report["trainable"])
    return model


def _render_charts(*args: Any, **kwargs: Any) -> None:
    """Lazy chart renderer injected into MetricsCallback (keeps `plots` out of `monitor`)."""
    from .plots import render_charts

    render_charts(*args, **kwargs)


def _make_callback(cfg: Config, run_dir: Path, *, ui: Rank0UI, world_size: int) -> MetricsCallback:
    """The rank-0 observability callback (charts render in a side thread)."""
    from .monitor import MetricsCallback

    return MetricsCallback(
        cfg,
        run_dir,
        console=ui.console,
        sampler=ui.sampler,
        writer=ui.writer,
        world_size=world_size,
        chart_renderer=_render_charts,
    )


def _build_trainer(
    cfg: Config,
    run_dir: Path,
    args: WorkerArgs,
    *,
    data: _TrainingData,
    model: PreTrainedModel | PeftModel,
    ui: Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> tuple[SFTTrainer, MetricsCallback | None]:
    """Phase 3: SFTConfig, observability callback, init-sync skip, SFTTrainer."""
    sft_config = build_sft_config(
        cfg, run_dir, world_size=world_size, n_train=data.n_train, dry_run=args.dry_run, max_steps=args.max_steps
    )
    if ui is not None:  # ui exists exactly on rank 0
        eff = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
        ui.console.print(
            f"[dim]effective batch: {cfg.optim.per_device_batch_size} (per GPU) × "
            f"{cfg.optim.grad_accum} (accum) × {world_size} (GPUs) = [bold]{eff}[/bold] sequences/step[/dim]"
        )

    metrics_callback = _make_callback(cfg, run_dir, ui=ui, world_size=world_size) if ui is not None else None
    callbacks = [metrics_callback] if metrics_callback is not None else []

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
    run_dir: Path,
    args: WorkerArgs,
    trainer: SFTTrainer,
    *,
    data: _TrainingData,
    ui: Rank0UI | None,
    world_size: int,
    logger: logging.Logger,
) -> Path:
    """Phase 4: train (resumable), save adapter + inference template."""
    import torch

    if args.resume:
        logger.info("resuming from checkpoint: %s", args.resume)
    trainer.train(resume_from_checkpoint=args.resume)

    if ui is not None:  # ui exists exactly on rank 0
        ui.console.print("[dim]saving adapter…[/dim]")
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


def _report_failure(run_dir: Path, writer: MetricsWriter, error: Exception) -> None:
    """Rank-0 abort path: record the abort event, print the failure lines."""
    from rich.console import Console

    from .monitor import EventRecord

    writer.write(EventRecord(type="event", level="error", event="abort", error=str(error)))
    writer.close()
    Console().print(
        f"[bold red]run aborted:[/bold red] {error}\n"
        f"  logs: {run_dir / 'logs'}\n"
        "  troubleshooting: docs/05-troubleshooting.md"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Worker entry: data → model → trainer → train → finalize (rank 0 owns the UI)."""
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    cfg = Config.from_toml(run_dir / "config.resolved.toml")

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
    ui = _rank0_ui(cfg, run_dir, enabled=is_main)

    try:
        data = _load_training_data(cfg, args, ui=ui, logger=logger)
        model = _build_model(cfg, local_rank, ui=ui, world_size=world_size, logger=logger)
        trainer, metrics_callback = _build_trainer(
            cfg, run_dir, args, data=data, model=model, ui=ui, world_size=world_size, logger=logger
        )
        adapter_dir = _train_and_save(run_dir, args, trainer, data=data, ui=ui, world_size=world_size, logger=logger)

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
            _report_failure(run_dir, ui.writer, exc)
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
    ui: Rank0UI,
    metrics_callback: MetricsCallback | None,
    *,
    start_time: float,
    world_size: int,
    dry_run: bool,
) -> None:
    """Rank-0 end of run: run_end event, charts, report, wrap-up."""
    from .monitor import RunEndRecord
    from .plots import render_charts, write_report

    records = metrics_callback.records if metrics_callback else []
    summary = _assemble_summary(adapter_dir, records, start_time=start_time)
    run_end: RunEndRecord = {
        "type": "run_end",
        "total_seconds": summary.total_seconds,
        "steps": summary.steps,
        "best_loss": summary.best_loss,
        "best_eval_loss": summary.best_eval_loss,
        "tokens_seen": summary.tokens_seen,
        "world_size": world_size,
        "dry_run": dry_run,
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

    _print_wrapup(ui.console, run_dir, summary, dry_run=dry_run)


def _assemble_summary(adapter_dir: Path, records: list[MetricsRecord], *, start_time: float) -> RunSummary:
    """End-of-run numbers for the wrap-up lines and the run_end record.

    Everything but the wall clock and the adapter path is derived from the
    metrics stream (see ``monitor.summarize``) — there is no second file to
    keep in sync.
    """
    from .monitor import summarize

    summary = summarize(records)
    summary.total_seconds = time.monotonic() - start_time
    summary.adapter_dir = str(adapter_dir)
    return summary


if __name__ == "__main__":
    main()
