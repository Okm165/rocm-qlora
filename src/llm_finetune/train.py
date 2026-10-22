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
from pathlib import Path

from .config import Config
from .utils import fmt_int, human_duration, seed_everything, write_json

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="llm-finetune training worker")
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
    log_file = run_dir / "logs" / f"rank-{rank}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llm_finetune")
    logger.setLevel(logging.INFO)
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


def ensure_training_chat_template(tokenizer, cfg: Config, log) -> str | None:
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


def _skip_redundant_init_sync(log) -> None:
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
        ddp_mod._sync_module_states = lambda *args, **kwargs: None
        log("DDP init broadcast disabled (identical checkpoints + shared seed → replicas identical by construction)")


def _install_grad_finite_checks(model, log) -> None:
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
        def hook(grad):
            if grad is not None and not bool(torch.isfinite(grad).all()):
                raise RuntimeError(
                    f"non-finite gradient in '{param_name}' — aborting before the optimizer step "
                    "(LLM_FINETUNE_GRAD_CHECK=1). See docs/05-troubleshooting.md"
                )

        return hook

    n = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_post_accumulate_grad_hook(make_hook(name))
            n += 1
    log("gradient finiteness checks installed on %d trainable tensors", n)


def build_sft_config(
    cfg: Config, run_dir: Path, *, world_size: int, n_train: int, dry_run: bool, max_steps: int | None
):
    from trl import SFTConfig

    eval_enabled = cfg.eval.per_epoch and cfg.data.eval_fraction > 0 and not dry_run
    # transformers v5 removed warmup_ratio — compute warmup steps ourselves
    # from the (estimated) total optimizer steps.
    effective_batch = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
    steps_per_epoch = max(1, -(-n_train // effective_batch))  # ceil division
    total_steps = max_steps or int(steps_per_epoch * cfg.optim.epochs)
    warmup_steps = max(1, int(total_steps * cfg.optim.warmup_ratio)) if cfg.optim.warmup_ratio else 0

    kwargs: dict = {
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
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
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
        "save_strategy": "epoch" if not dry_run else "no",
        # Full checkpoints (adapter + optimizer/scheduler/rng state, ~0.5 GB
        # each at 27B) so an aborted run can truly resume with
        # `llm-finetune train --resume <run-dir>`.
        "save_only_model": False,
        "save_total_limit": None if cfg.logging.keep_epoch_checkpoints else 1,
        "eval_strategy": "epoch" if eval_enabled else "no",
        "report_to": [],
        "disable_tqdm": True,
        "log_level": "warning",
        "seed": cfg.run.seed,
        "data_seed": cfg.run.seed,
        "dataloader_num_workers": min(8, (os.cpu_count() or 8) // max(1, world_size)),
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "torch_empty_cache_steps": 50,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if dry_run:
        kwargs["max_steps"] = max_steps or 3

    # `include_num_input_tokens_seen` changed type across transformers
    # versions (bool -> str enum with values all|non_padding|no).
    try:
        field = SFTConfig.__dataclass_fields__.get("include_num_input_tokens_seen")
        default = getattr(field, "default", None)
        kwargs["include_num_input_tokens_seen"] = "non_padding" if isinstance(default, str) else True
    except Exception:
        pass

    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
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

    # -- rank-0 terminal -----------------------------------------------------
    console = None
    book = None
    dashboard = None
    sampler = None
    writer = None
    metrics_callback = None
    start_time = time.monotonic()
    if is_main:
        from rich.console import Console

        from .explain import ExplainerBook
        from .monitor import GpuSampler, MetricsCallback, MetricsWriter, RichDashboard

        console = Console()
        book = ExplainerBook(console, enabled=cfg.run.explainers)
        sampler = GpuSampler(cfg.logging.gpu_sample_seconds)
        writer = MetricsWriter(run_dir / "metrics.jsonl")
        dashboard = RichDashboard(console, refresh_per_second=cfg.run.console_refresh_per_second)
        dashboard.state.run_name = run_dir.name
        dashboard.state.model = cfg.model.name.split("/")[-1]
        dashboard.state.world_size = world_size
        dashboard.update(phase="loading data")

    try:
        # ------------------------------------------------------------------
        # 1. Data
        # ------------------------------------------------------------------
        from .data import load_sharegpt, profile_tokens, render_preview, split_dataset, to_hf_dataset
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

        if is_main:
            from rich.panel import Panel
            from rich.table import Table

            pcts = profile.percentiles()
            table = Table.grid(padding=(0, 2))
            table.add_column(style="dim", justify="right")
            table.add_column()
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

        # ------------------------------------------------------------------
        # 2. Model
        # ------------------------------------------------------------------
        if is_main:
            dashboard.update(phase="loading model")
            book.show("prepared_checkpoint")
            book.show("hybrid_attention")
            book.show("rocm_attention")

        from .launcher import resolve_attn_impl

        attn = resolve_attn_impl(cfg)
        from .model import attach_lora, load_model_for_training, parameter_report, prepare_for_qlora

        model = load_model_for_training(cfg, local_rank=local_rank, attn=attn)
        model = prepare_for_qlora(model, cfg)
        model = attach_lora(model, cfg)
        report = parameter_report(model)

        # --- optional numerical diagnostics (env-gated) ----------------------
        # Both exist because of a real failure mode: a run can hit a step with
        # a finite loss but NaN gradients; DDP then spreads the NaN to every
        # rank, the optimizer poisons the weights, and from then on the fused
        # loss reports exactly 0.0 (zero on ignored positions) while the EMA
        # keeps "improving". The on_log guard aborts, but *these* pinpoint.
        if os.environ.get("LLM_FINETUNE_DETECT_ANOMALY") == "1":
            torch.autograd.set_detect_anomaly(True)
            logger.info("autograd anomaly detection ON — the first NaN-producing op will raise with a traceback")
        if os.environ.get("LLM_FINETUNE_GRAD_CHECK") == "1":
            _install_grad_finite_checks(model, logger.info)

        # Reclaim the load-time casting transients still sitting in the
        # allocator's reserved pool (~2.4 GB for this model). Without this,
        # 27B training on 24 GB cards starts with almost no headroom.
        torch.cuda.empty_cache()

        if is_main:
            from rich.panel import Panel
            from rich.table import Table

            table = Table.grid(padding=(0, 2))
            table.add_column(style="dim", justify="right")
            table.add_column()
            table.add_row("total params", fmt_int(report["total"]))
            table.add_row("trainable (LoRA)", f"{fmt_int(report['trainable'])}  ({report['pct']:.3f}%)")
            breakdown = " · ".join(f"{k}: {fmt_int(v)}" for k, v in list(report["by_suffix"].items())[:8])
            table.add_row("by module", breakdown)
            table.add_row("attention", f"{attn}  (doctor-verified where available)")
            console.print(Panel(table, title="[bold]model[/bold]", border_style="cyan"))
            book.show("lora_targets")
            book.show("rank_alpha")
            book.show("assistant_only_loss")
            book.show("chunked_loss")
            book.show("multi_gpu_ddp")
            if writer:
                writer.write(
                    {
                        "type": "event",
                        "level": "info",
                        "event": "model_loaded",
                        "total_params": report["total"],
                        "trainable": report["trainable"],
                        "attn": attn,
                        "world_size": world_size,
                    }
                )
        logger.info("model ready: %d trainable params", report["trainable"])

        # ------------------------------------------------------------------
        # 3. Trainer
        # ------------------------------------------------------------------
        if is_main:
            dashboard.update(phase="building trainer")
            book.show("effective_batch")
            book.show("warmup")
            book.show("vram_budget")

        sft_config = build_sft_config(
            cfg, run_dir, world_size=world_size, n_train=n_train, dry_run=args.dry_run, max_steps=args.max_steps
        )
        if is_main and console:
            eff = cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size
            console.print(
                f"[dim]effective batch: {cfg.optim.per_device_batch_size} (per GPU) × "
                f"{cfg.optim.grad_accum} (accum) × {world_size} (GPUs) = [bold]{eff}[/bold] sequences/step[/dim]"
            )

        callbacks = []
        if is_main:
            from .monitor import MetricsCallback

            metrics_callback = MetricsCallback(
                cfg,
                run_dir,
                console=console,
                dashboard=dashboard,
                book=book,
                sampler=sampler,
                writer=writer,
                total_tokens_estimate=total_tokens_estimate,
                world_size=world_size,
            )
            callbacks = [metrics_callback]

        if world_size > 1:
            _skip_redundant_init_sync(logger.info)

        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # ------------------------------------------------------------------
        # 4. Train
        # ------------------------------------------------------------------
        if is_main:
            dashboard.update(phase="training")
            dashboard.start()
            book.show("reading_loss")
            book.show("grad_norm")
            book.show("gpu_health")
            if eval_ds is not None:
                book.show("eval")
            book.hint(f"charts refresh every {cfg.logging.chart_snapshot_steps} steps into {run_dir / 'charts'}")

        if args.resume:
            logger.info("resuming from checkpoint: %s", args.resume)
        trainer.train(resume_from_checkpoint=args.resume)

        if is_main:
            dashboard.update(phase="saving adapter", step=trainer.state.global_step)
        adapter_dir = run_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        # Save the *inference* template with the adapter (the training template
        # with generation markers is an implementation detail of the loss mask).
        tokenizer.chat_template = original_chat_template
        tokenizer.save_pretrained(adapter_dir)
        logger.info("adapter saved to %s", adapter_dir)

        if world_size > 1:
            with contextlib.suppress(Exception):
                torch.distributed.barrier()

        # ------------------------------------------------------------------
        # 5. Finalize (rank 0)
        # ------------------------------------------------------------------
        if is_main:
            dashboard.update(phase="finalizing")
            dashboard.stop()

            from .monitor import summarize
            from .plots import render_charts, write_report

            records = metrics_callback.records if metrics_callback else []
            summary = summarize(records)
            summary.total_seconds = time.monotonic() - start_time
            summary_dict = summary.to_dict()
            summary_dict.update(
                {
                    "run_dir": str(run_dir),
                    "adapter_dir": str(adapter_dir),
                    "model": cfg.model.name,
                    "dataset": cfg.data.path,
                    "effective_batch": cfg.optim.per_device_batch_size * cfg.optim.grad_accum * world_size,
                    "world_size": world_size,
                    "max_length": cfg.data.max_length,
                    "lora_r": cfg.lora.r,
                    "lr": cfg.optim.lr,
                    "epochs": cfg.optim.epochs,
                    "loss_type": cfg.optim.loss_type,
                    "dry_run": args.dry_run,
                }
            )
            write_json(run_dir / "summary.json", summary_dict)
            if writer:
                writer.write(
                    {
                        "type": "run_end",
                        "total_seconds": summary.total_seconds,
                        **{k: summary_dict[k] for k in ("steps", "best_loss", "best_eval_loss", "tokens_seen")},
                    }
                )
                writer.close()

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

            _print_wrapup(console, book, cfg, run_dir, summary_dict, dry_run=args.dry_run)

        if world_size > 1:
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()

    except Exception as exc:
        logger.error("training failed: %s\n%s", exc, traceback.format_exc())
        if is_main:
            if dashboard:
                with contextlib.suppress(Exception):
                    dashboard.stop()
            if writer:
                writer.write({"type": "event", "level": "error", "event": "abort", "error": str(exc)})
                writer.close()
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold red]Run aborted:[/bold red] {exc}\n\n"
                    f"Full logs: {run_dir / 'logs'}\n"
                    "Troubleshooting guide: docs/05-troubleshooting.md",
                    title="[red]failure[/red]",
                    border_style="red",
                )
            )
        raise


# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------


def _print_wrapup(console, book, cfg: Config, run_dir: Path, summary: dict, *, dry_run: bool) -> None:
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("steps", fmt_int(summary.get("steps")))
    table.add_row(
        "loss",
        f"{summary.get('first_loss', float('nan')):.4f} → {summary.get('last_loss', float('nan')):.4f} "
        f"(best {summary.get('best_loss', float('nan')):.4f} @ step {summary.get('best_step')})",
    )
    if summary.get("last_eval_loss") is not None:
        table.add_row(
            "eval loss",
            f"{summary['first_eval_loss']:.4f} → {summary['last_eval_loss']:.4f} "
            f"(best {summary['best_eval_loss']:.4f})",
        )
    table.add_row("tokens seen", fmt_int(summary.get("tokens_seen")))
    table.add_row("wall time", human_duration(summary.get("total_seconds")))
    table.add_row("peak VRAM (max GPU)", f"{summary.get('peak_vram_gb', float('nan')):.1f} GB")
    table.add_row("adapter", str(Path(summary.get("adapter_dir", ""))))
    table.add_row("charts", str(run_dir / "charts"))
    table.add_row("report", str(run_dir / "report.md"))
    console.print(Panel(table, title="[bold green]run complete[/bold green]", border_style="green"))
    if dry_run:
        console.print("[yellow]This was a dry run (3 steps). Launch without --dry-run for the real thing.[/yellow]")
    book.show("next_steps")
    console.print(
        "[dim]next:[/dim] "
        f"[bold]llm-finetune chat --run {run_dir}[/bold]  ·  "
        f"[bold]llm-finetune merge --run {run_dir}[/bold]  ·  "
        f"[bold]llm-finetune report --run {run_dir}[/bold]"
    )


if __name__ == "__main__":
    main()
