"""`rocm-qlora <command>` — the project's entry point.

Commands
--------
doctor    environment + hardware report; SDPA/bnb probes; writes .context/
data      dataset profile: token lengths, coverage, preview, chart
prepare   quantize/surgery the base model into .data/models/ (once)
train     preflight + launch (single GPU or DDP via torchrun)
report    regenerate charts + Markdown/HTML report from metrics.jsonl
chat      interactive sanity check against a trained adapter
merge     merge adapter into a bf16 model (ready for GGUF export)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, ConfigError
from .doctor import cmd_doctor
from .utils import fmt_int, human_bytes, human_duration, timestamp_slug

if TYPE_CHECKING:
    from .data import DatasetStats, Percentiles, TokenProfile

DEFAULT_CONFIG = "configs/qwen3.8-27b.toml"
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_toml(args.config)
    overrides: dict[str, object] = {}
    if getattr(args, "gpus", None):
        if args.gpus != "auto":
            overrides["dist.gpus"] = [int(g) for g in str(args.gpus).split(",")]
        else:
            overrides["dist.gpus"] = "auto"
    if getattr(args, "dist", None):
        overrides["dist.mode"] = args.dist
    if getattr(args, "run_name", None):
        overrides["run.name"] = args.run_name
    if getattr(args, "seed", None) is not None:
        overrides["run.seed"] = args.seed
    if getattr(args, "max_length", None):
        overrides["data.max_length"] = args.max_length
    for item in getattr(args, "set", None) or []:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key.strip()] = value
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def _resolve_resume_checkpoint(path: str) -> Path | None:
    """Resolve a run dir / adapter dir / checkpoint dir to the latest checkpoint."""
    p = Path(path)
    if not p.exists():
        return None
    if (p / "trainer_state.json").exists():
        return p  # already a checkpoint directory
    candidates = sorted(p.glob("adapter/checkpoint-*")) or sorted(p.glob("checkpoint-*"))
    if not candidates:
        return None
    return max(candidates, key=lambda c: int(c.name.rsplit("-", 1)[-1]))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _print_dataset_table(cfg: Config, stats: DatasetStats) -> None:
    """Raw-file summary table (examples, turns, roles, char percentiles)."""
    table = Table(title=f"{cfg.data.path}")
    table.add_column("property", style="dim")
    table.add_column("value")
    table.add_row("examples", fmt_int(stats.n_examples))
    table.add_row("turns/example", ", ".join(f"{k}: {v}" for k, v in stats.turns_per_example.items()))
    table.add_row("roles", ", ".join(f"{k}: {v}" for k, v in stats.role_counts.items()))
    cp = stats.char_percentiles()
    table.add_row(
        "assistant chars",
        f"p50={cp.p50:.0f} p90={cp.p90:.0f} p95={cp.p95:.0f} p99={cp.p99:.0f} p100={cp.p100:.0f}",
    )
    table.add_row("reasoning traces", f"{stats.think_blocks} with  thinking, {stats.think_complete} complete")
    console.print(table)


def _print_token_table(cfg: Config, profile: TokenProfile, pcts: Percentiles) -> None:
    """Chat-templated token-length table."""
    ttable = Table(title="token lengths (chat-templated)")
    ttable.add_column("metric", style="dim")
    ttable.add_column("value", justify="right")
    for name, value in asdict(pcts).items():
        ttable.add_row(name, f"{value:.0f}")
    ttable.add_row("total tokens", fmt_int(profile.total_tokens))
    ttable.add_row(
        f"coverage @ max_length={cfg.data.max_length}", f"{profile.coverage_at(cfg.data.max_length) * 100:.1f}%"
    )
    ttable.add_row("truncated", f"{profile.n_truncated} ({profile.truncation_pct:.1f}%)")
    console.print(ttable)


def _print_max_length_hint(cfg: Config, pcts: Percentiles) -> None:
    """Suggest a max_length at p99 (rounded to 128)."""
    suggestion = int(round(pcts.p99 / 128) * 128)
    console.print(
        f"[dim]hint: for ≥99% coverage set data.max_length ≈ [bold]{suggestion}[/bold] "
        f"(current: {cfg.data.max_length})[/dim]"
    )


def _write_profile_chart(cfg: Config, profile: TokenProfile) -> Path:
    """Token-length histogram under <output_root>/data-profile/<timestamp>."""
    from .plots import token_length_hist

    out_dir = Path(cfg.run.output_root) / "data-profile" / timestamp_slug()
    out_dir.mkdir(parents=True, exist_ok=True)
    token_length_hist(
        profile.lengths,
        out_dir / "token_lengths.png",
        max_length=cfg.data.max_length,
        meta={"run_name": "data-profile"},
    )
    return out_dir


def cmd_data(args: argparse.Namespace) -> int:
    """Profile the dataset (tokens, coverage, truncation) and plot the histogram."""
    from .data import load_sharegpt, profile_tokens, render_preview
    from .model import load_tokenizer

    cfg = _load_config(args)
    console.rule("[bold cyan]dataset profile[/bold cyan]")
    examples, stats = load_sharegpt(cfg.data.path, cfg.data.messages_field)

    _print_dataset_table(cfg, stats)

    console.print("[dim]tokenizing with the model's chat template…[/dim]")
    tokenizer = load_tokenizer(cfg)
    profile = profile_tokens(
        tokenizer, examples, messages_field=cfg.data.messages_field, max_length=cfg.data.max_length
    )
    pcts = profile.percentiles()
    _print_token_table(cfg, profile, pcts)
    _print_max_length_hint(cfg, pcts)

    out_dir = _write_profile_chart(cfg, profile)

    console.print(
        Panel(
            render_preview(tokenizer, examples[0], messages_field=cfg.data.messages_field),
            title="first example",
            border_style="blue",
        )
    )
    console.print(f"[dim]saved: {out_dir}[/dim]")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Create the prepared NF4 checkpoint from the source model."""
    from .model import prepare_checkpoint

    cfg = _load_config(args)
    console.rule("[bold cyan]prepare checkpoint[/bold cyan]")
    console.print(
        f"source: [bold]{cfg.model.name}[/bold] → target: [bold]{cfg.model.prepared_dir}[/bold]\n"
        f"[dim]requantize: {cfg.quantization.requantize_modules} · vision: "
        f"{'quantized' if cfg.quantization.requantize_vision else 'as-is'}[/dim]"
    )
    result = prepare_checkpoint(cfg, force=args.force, log=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    table = Table(title="prepared checkpoint")
    table.add_column("property", style="dim")
    table.add_column("value")
    table.add_row("path", str(result.out_dir))
    table.add_row("total params", f"{result.total_params / 1e9:.2f} B")
    table.add_row("re-quantized modules", str(len(result.requantized)))
    table.add_row("disk size", human_bytes(result.disk_gb * 1e9))
    if result.seconds:
        table.add_row("prepare time", human_duration(result.seconds))
    console.print(table)
    console.print("[dim]next: uv run rocm-qlora train --config …[/dim]")
    return 0


def _ensure_prepared(cfg: Config, args: argparse.Namespace) -> bool:
    """Print the prepare-required panel when the NF4 checkpoint is missing."""
    prepared = Path(cfg.model.prepared_dir)
    if (prepared / "config.json").exists():
        return True
    console.print(
        Panel(
            f"The prepared checkpoint [bold]{prepared}[/bold] does not exist yet.\n\n"
            f"Run first:  [bold]uv run rocm-qlora prepare --config {args.config}[/bold]\n\n"
            "[dim]docs/01 + docs/02 explain what prepare does and why.[/dim]",
            title="[yellow]prepare required[/yellow]",
            border_style="yellow",
        )
    )
    return False


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    """Resolve --resume to a checkpoint dir; None when absent or not found."""
    if not args.resume:
        return None
    checkpoint = _resolve_resume_checkpoint(args.resume)
    if checkpoint is None:
        console.print(f"[red]no checkpoint found under {args.resume}[/red]")
        return None
    console.print(f"[bold]resuming from[/bold] {checkpoint}")
    return checkpoint


def _worker_extra_args(args: argparse.Namespace, resume_checkpoint: Path | None) -> list[str]:
    """CLI flags forwarded to the training worker."""
    extra = ["--dry-run"] if args.dry_run else []
    if args.max_steps:
        extra += ["--max-steps", str(args.max_steps)]
    if resume_checkpoint:
        extra += ["--resume", str(resume_checkpoint)]
    return extra


def _print_launch_failure(run_dir: Path, returncode: int) -> None:
    """Failure panel for a non-zero worker exit."""
    console.print(
        Panel(
            f"training exited with code {returncode}\nlogs: {run_dir / 'logs'} · see docs/05-troubleshooting.md",
            title="[red]failure[/red]",
            border_style="red",
        )
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Preflight GPUs, snapshot the config, and launch the training worker(s)."""
    from .launcher import launch_worker, make_run_dir, preflight_gpus, write_env_fingerprint

    cfg = _load_config(args)

    console.rule(f"[bold cyan]train[/bold cyan] · {cfg.run.name}")
    if not _ensure_prepared(cfg, args):
        return 2

    plan = preflight_gpus(cfg)
    for warning in plan.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    console.print(
        f"[bold]GPUs for this run:[/bold] {plan.gpus} "
        f"(free: {', '.join(f'{g}: {plan.free_gb.get(g, float("nan")):.1f} GB' for g in plan.gpus)})"
    )
    console.print(
        "[dim]fit is not predicted here — the run fits or OOMs; "
        "docs/02 has the measured 24 GB budget, docs/05 the knobs to react with.[/dim]"
    )

    if args.dry_run:
        console.print("[yellow]dry run: 3 steps, 8 examples, no eval — the full pipeline in miniature[/yellow]")

    run_dir = make_run_dir(cfg, dry_run=args.dry_run)
    cfg.snapshot(run_dir)
    write_env_fingerprint(run_dir, cfg, plan.gpus)
    console.print(f"[dim]run directory: {run_dir}[/dim]")

    resume_checkpoint = _resolve_resume(args)
    if args.resume and resume_checkpoint is None:
        return 2

    code = launch_worker(cfg, run_dir, plan.gpus, extra_args=_worker_extra_args(args, resume_checkpoint))
    if code != 0:
        _print_launch_failure(run_dir, code)
    return code


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate charts and the Markdown report for a finished run."""
    from .plots import replot, write_report

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")
    console.rule("[bold cyan]report[/bold cyan]")
    charts = replot(run_dir)
    report = write_report(run_dir)
    console.print(f"[green]✓[/green] regenerated {len(charts)} chart(s) in {run_dir / 'charts'}")
    console.print(f"[green]✓[/green] report: [bold]{report}[/bold]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Talk to a trained adapter (thin wrapper around chat.run_chat)."""
    from .chat import run_chat

    console.rule("[bold cyan]chat[/bold cyan]")
    run_chat(
        run_dir=args.run,
        base=args.base,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        console=console,
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Fold the adapter into the base model in bf16 (ready for GGUF export)."""
    from .merge import merge_run

    console.rule("[bold cyan]merge[/bold cyan]")
    merge_run(args.run, out_dir=args.out, console=console)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The full CLI: one subparser per command; shared flags via add_common."""
    parser = argparse.ArgumentParser(
        prog="rocm-qlora",
        description="Teaching-first QLoRA fine-tuning for AMD ROCm GPUs",
    )
    parser.add_argument("--version", action="version", version=f"rocm-qlora {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        """Attach the flags shared by every subcommand."""
        p.add_argument("--config", default=DEFAULT_CONFIG, help=f"TOML config (default: {DEFAULT_CONFIG})")
        p.add_argument(
            "--set", action="append", metavar="KEY=VALUE", help="arbitrary config override, e.g. --set optim.lr=2e-4"
        )

    p = sub.add_parser("doctor", help="environment + hardware report")
    p.add_argument("--fast", action="store_true", help="skip GPU probes (SDPA, bnb)")
    p.add_argument("--bench", action="store_true", help="also run the bf16 matmul benchmark")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("data", help="profile the dataset")
    add_common(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("prepare", help="quantize/surgery the base model (once)")
    add_common(p)
    p.add_argument("--force", action="store_true", help="rebuild even if the prepared dir exists")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="run QLoRA training")
    add_common(p)
    p.add_argument("--gpus", default=None, help="'auto' (default) or comma-separated ids, e.g. 0,2")
    p.add_argument("--dist", choices=["auto", "single", "ddp"], default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="3 steps on 8 examples — validate the pipeline")
    p.add_argument(
        "--resume",
        default=None,
        metavar="RUN_OR_CHECKPOINT",
        help="resume an aborted run: pass its run dir (or a checkpoint dir); "
        "the latest checkpoint is picked automatically",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="regenerate charts + report for a run")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("chat", help="interactive chat with a trained adapter")
    p.add_argument("--run", default=None, help="run directory (adapter + config are read from it)")
    p.add_argument("--base", default=None, help="override base model (prepared dir)")
    p.add_argument("--adapter", default=None, help="override adapter directory")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("merge", help="merge adapter into a bf16 model")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_merge)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Point the torch wheel's libdrm at the system ASIC-ID table (silences a
    # per-GPU stderr flood at HIP init; workers inherit it via build_worker_env).
    """Parse arguments and dispatch to the selected command."""
    import os

    os.environ.setdefault("AMDGPU_ASIC_ID_TABLE_PATHS", "/usr/share/libdrm")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        console.print(Panel(str(exc), title="[red]configuration error[/red]", border_style="red"))
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted — run artifacts were preserved in .outputs/[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
