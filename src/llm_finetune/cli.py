"""`llm-finetune <command>` — the project's entry point.

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
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, ConfigError
from .utils import fmt_int, human_bytes, human_duration, timestamp_slug, write_json

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
    if getattr(args, "quiet", False):
        overrides["run.explainers"] = False
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


def _load_geometry(cfg: Config):
    """Model geometry: exact from prepare's meta, else fetched config.json."""
    from .hardware import geometry_from_finetune_meta, geometry_from_hf_config

    meta = Path(cfg.model.prepared_dir) / "finetune_meta.json"
    if meta.exists():
        geom = geometry_from_finetune_meta(meta, lora_r=cfg.lora.r)
        if geom:
            return geom
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(cfg.model.name, "config.json")
        return geometry_from_hf_config(Path(p), lora_r=cfg.lora.r)
    except Exception:
        return None


def _print_vram_estimate(cfg: Config, geom) -> float:
    """Print the VRAM decomposition table; returns the estimated total (GB)."""
    from .hardware import estimate_training_vram

    est = estimate_training_vram(
        geom,
        lora_r=cfg.lora.r,
        max_length=cfg.data.max_length,
        batch_size=cfg.optim.per_device_batch_size,
        grad_checkpointing=cfg.optim.gradient_checkpointing,
        optimizer=cfg.optim.optimizer,
        loss_type=cfg.optim.loss_type,
    )
    table = Table(title="VRAM estimate (per GPU, one training process)", show_lines=False)
    table.add_column("component", style="dim")
    table.add_column("GB", justify="right")
    for name, value in est.breakdown().items():
        style = "bold" if name == "TOTAL" else ""
        table.add_row(name, f"{value:.1f}", style=style)
    console.print(table)
    for note in est.notes:
        console.print(f"[dim]· {note}[/dim]")
    return est.total_gb


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


def cmd_doctor(args: argparse.Namespace) -> int:
    from .hardware import (
        cpu_info,
        disk_free_gb,
        gpu_snapshot,
        host_memory,
        kernel_version,
        matmul_benchmark,
        package_versions,
        probe_sdpa,
        rocm_driver_version,
        torch_runtime_info,
    )

    _load_config(args) if getattr(args, "config", None) else None
    context_dir = Path(".context")
    context_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]llm-finetune doctor[/bold cyan] · v{__version__}")

    # --- host ---------------------------------------------------------------
    mem = host_memory()
    cpu = cpu_info()
    host = Table.grid(padding=(0, 2))
    host.add_column(style="dim", justify="right")
    host.add_column()
    host.add_row("CPU", f"{cpu.get('model', '?')} ({cpu.get('cores')} threads)")
    host.add_row("RAM", f"{mem['total_gib']:.0f} GiB total · {mem['available_gib']:.0f} GiB available")
    host.add_row("kernel", kernel_version())
    host.add_row("amdgpu driver", rocm_driver_version() or "n/a")
    host.add_row("disk free", human_bytes(disk_free_gb(".") * 1e9))
    console.print(Panel(host, title="host", border_style="blue"))

    # --- GPUs ---------------------------------------------------------------
    gpus = gpu_snapshot()
    gpu_table = Table(title="GPUs (rocm-smi)")
    for col in ("#", "VRAM total", "VRAM free", "temp", "power", "util"):
        gpu_table.add_column(col, justify="right" if col != "#" else "left")
    for g in gpus:
        gpu_table.add_row(
            str(g.index),
            f"{g.mem_total_b / 1e9:.0f} GB",
            f"{g.mem_free_gb:.1f} GB",
            f"{g.temp_edge_c:.0f}°C" if g.temp_edge_c is not None else "?",
            f"{g.power_w:.0f} W" if g.power_w is not None else "?",
            f"{g.util_pct:.0f}%" if g.util_pct is not None else "?",
        )
    console.print(gpu_table)

    # --- versions -----------------------------------------------------------
    versions = package_versions()
    vtable = Table(title="software")
    vtable.add_column("package", style="dim")
    vtable.add_column("version")
    for name, version in versions.items():
        vtable.add_row(name, version)
    console.print(vtable)

    # --- torch runtime ------------------------------------------------------
    runtime = torch_runtime_info()
    rtable = Table(title="torch runtime")
    rtable.add_column("key", style="dim")
    rtable.add_column("value")
    for key, value in runtime.items():
        rtable.add_row(key, str(value))
    console.print(rtable)

    # --- SDPA probe ---------------------------------------------------------
    verdict: dict = {}
    if not args.fast:
        console.print("[dim]probing SDPA backends (fresh subprocess per backend, ~30-60 s)…[/dim]")
        sdpa = probe_sdpa()
        verdict["sdpa"] = sdpa
        stable = Table(title="SDPA backend probe")
        stable.add_column("backend", style="dim")
        stable.add_column("AOTriton env on", justify="center")
        stable.add_column("env off", justify="center")
        for backend in ("flash", "efficient", "math"):
            on = sdpa["backends"].get(f"{backend}_on", {})
            off = sdpa["backends"].get(f"{backend}_off", {})
            stable.add_row(
                backend,
                "[green]✓[/green]" if on.get("ok") else f"[red]✗[/red] [dim]{str(on.get('error', ''))[:40]}[/dim]",
                "[green]✓[/green]" if off.get("ok") else "[red]✗[/red]",
            )
        console.print(stable)
        console.print(f"[bold]verdict:[/bold] {sdpa['verdict']} → attn_implementation={sdpa['attn_implementation']}")

    # --- bitsandbytes 4-bit smoke test ---------------------------------------
    if not args.fast:
        bnb_result = _bnb_smoke()
        verdict["bnb_smoke"] = bnb_result
        status = "[green]✓[/green]" if bnb_result.get("ok") else f"[red]✗[/red] {bnb_result.get('error', '')}"
        console.print(f"bitsandbytes 4-bit smoke (train + decode shapes): {status}")
        if bnb_result.get("decode_nan"):
            console.print(
                "[red]decode-shape NaN detected — upgrade bitsandbytes (ROCm GEMV fix) before using `chat`[/red]"
            )

    # --- RCCL IPC transport probe --------------------------------------------
    if not args.fast:
        from .hardware import probe_hip_ipc

        ipc = probe_hip_ipc()
        verdict["hip_ipc"] = ipc
        if ipc.get("ipc_ok"):
            console.print(
                "RCCL IPC handles (hipIpcGetMemHandle, dma-buf mode): [green]✓[/green] → P2P collectives enabled"
            )
        else:
            console.print(
                f"RCCL IPC handles: [red]✗[/red] {ipc.get('error', '')} — "
                "training will fall back to shared-memory collectives; set NCCL_P2P_DISABLE=1 in [env.extra] to silence"
            )

    # --- optional matmul benchmark -------------------------------------------
    if args.bench:
        console.print("[dim]bf16 matmul benchmark (8192³)…[/dim]")
        bench = matmul_benchmark()
        verdict["bench"] = bench
        console.print(
            f"[bold]{bench['tflops']:.1f} TFLOPS[/bold] bf16 (8192³, {bench['seconds'] * 1000:.0f} ms/matmul)"
        )

    # --- write artifacts ------------------------------------------------------
    verdict["created"] = timestamp_slug()
    verdict["host"] = {"ram": mem, "cpu": cpu}
    verdict["packages"] = versions
    write_json(context_dir / "doctor-latest.json", verdict)
    md = _doctor_markdown(verdict, runtime, gpus)
    (context_dir / "doctor-latest.md").write_text(md)
    console.print(f"[dim]saved: {context_dir / 'doctor-latest.json'} · {context_dir / 'doctor-latest.md'}[/dim]")
    console.print(
        "[dim]hint: the train launcher reads the SDPA verdict from doctor-latest.json to pick "
        "attention settings automatically.[/dim]"
    )
    return 0


def _bnb_smoke() -> dict:
    """Quantize a Linear on GPU; check train-shape and decode-shape numerics."""
    import torch

    try:
        import bitsandbytes as bnb

        from .hardware import gpu_snapshot

        snap = gpu_snapshot()
        device = f"cuda:{max(snap, key=lambda g: g.mem_free_b).index}" if snap else "cuda:0"
        torch.manual_seed(0)
        lin = torch.nn.Linear(512, 256, bias=False).to(device, dtype=torch.float16)
        q = bnb.nn.Linear4bit(
            512, 256, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4"
        )
        q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False, quant_type="nf4", compress_statistics=True)
        q = q.to(device)
        result: dict = {"ok": True}
        for seq in (1, 4):
            x = torch.randn(1, seq, 512, device=device, dtype=torch.bfloat16)
            y = q(x)
            ref = lin(x.to(torch.float16)).float()
            err = (y.float() - ref).abs().max().item()
            result[f"max_err_seq{seq}"] = err
            if seq == 1 and (torch.isnan(y).any() or err > 1.0):
                result["decode_nan"] = True
                result["ok"] = False
        return result
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _doctor_markdown(verdict: dict, runtime: dict, gpus) -> str:
    lines = [f"# Doctor report — {verdict.get('created', '')}", ""]
    lines.append(f"- torch: {runtime.get('torch', '?')} · HIP: {runtime.get('hip', '?')}")
    lines.append(f"- devices: {runtime.get('device_count', 0)}")
    lines.extend(f"- GPU {g.index}: {g.mem_total_b / 1e9:.0f} GB total, {g.mem_free_gb:.1f} GB free" for g in gpus)
    if "sdpa" in verdict:
        lines.append(
            f"- SDPA verdict: **{verdict['sdpa']['verdict']}** (attn={verdict['sdpa']['attn_implementation']})"
        )
    if "bnb_smoke" in verdict:
        lines.append(f"- bnb smoke: {verdict['bnb_smoke']}")
    if "bench" in verdict:
        lines.append(f"- bf16 matmul: {verdict['bench']['tflops']:.1f} TFLOPS")
    return "\n".join(lines) + "\n"


def cmd_data(args: argparse.Namespace) -> int:
    from .data import load_sharegpt, profile_tokens, render_preview
    from .model import load_tokenizer
    from .plots import token_length_hist

    cfg = _load_config(args)
    console.rule("[bold cyan]dataset profile[/bold cyan]")
    examples, stats = load_sharegpt(cfg.data.path, cfg.data.messages_field)

    table = Table(title=f"{cfg.data.path}")
    table.add_column("property", style="dim")
    table.add_column("value")
    table.add_row("examples", fmt_int(stats.n_examples))
    table.add_row("turns/example", ", ".join(f"{k}: {v}" for k, v in stats.turns_per_example.items()))
    table.add_row("roles", ", ".join(f"{k}: {v}" for k, v in stats.role_counts.items()))
    cp = stats.char_percentiles("assistant")
    table.add_row("assistant chars", " ".join(f"p{k[1:]}={v:.0f}" for k, v in cp.items()))
    table.add_row("reasoning traces", f"{stats.think_blocks} with  thinking, {stats.think_complete} complete")
    console.print(table)

    console.print("[dim]tokenizing with the model's chat template…[/dim]")
    tokenizer = load_tokenizer(cfg)
    profile = profile_tokens(
        tokenizer, examples, messages_field=cfg.data.messages_field, max_length=cfg.data.max_length
    )
    pcts = profile.percentiles()
    ttable = Table(title="token lengths (chat-templated)")
    ttable.add_column("metric", style="dim")
    ttable.add_column("value", justify="right")
    for k, v in pcts.items():
        ttable.add_row(k, f"{v:.0f}")
    ttable.add_row("total tokens", fmt_int(profile.total_tokens))
    coverage = profile.coverage_at(cfg.data.max_length)
    ttable.add_row(f"coverage @ max_length={cfg.data.max_length}", f"{coverage * 100:.1f}%")
    ttable.add_row("truncated", f"{profile.n_truncated} ({profile.truncation_pct:.1f}%)")
    console.print(ttable)

    # suggestion for max_length at p99
    p99 = pcts["p99"]
    suggestion = int(round(p99 / 128) * 128)
    console.print(
        f"[dim]hint: for ≥99% coverage set data.max_length ≈ [bold]{suggestion}[/bold] "
        f"(current: {cfg.data.max_length})[/dim]"
    )

    out_dir = Path(cfg.run.output_root) / "data-profile" / timestamp_slug()
    out_dir.mkdir(parents=True, exist_ok=True)
    token_length_hist(
        profile.lengths,
        out_dir / "token_lengths.png",
        max_length=cfg.data.max_length,
        meta={"run_name": "data-profile"},
    )
    write_json(
        out_dir / "profile.json",
        {
            "n_examples": stats.n_examples,
            "percentiles": pcts,
            "coverage_at_max_length": coverage,
            "max_length": cfg.data.max_length,
            "truncated": profile.n_truncated,
            "total_tokens": profile.total_tokens,
            "suggested_max_length_p99": suggestion,
            "think_blocks": stats.think_blocks,
        },
    )
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
    console.print("[dim]next: uv run llm-finetune train --config …[/dim]")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .explain import ExplainerBook
    from .launcher import launch_worker, make_run_dir, preflight_gpus, write_env_fingerprint

    cfg = _load_config(args)
    book = ExplainerBook(console, enabled=cfg.run.explainers)

    console.rule(f"[bold cyan]train[/bold cyan] · {cfg.run.name}")
    prepared = Path(cfg.model.prepared_dir)
    if not (prepared / "config.json").exists():
        console.print(
            Panel(
                f"The prepared checkpoint [bold]{prepared}[/bold] does not exist yet.\n\n"
                f"Run first:  [bold]uv run llm-finetune prepare --config {args.config}[/bold]\n\n"
                "[dim]Why: quantizing once to NF4 avoids staging the full bf16 model in host RAM "
                "in every data-parallel process (see the 'prepared checkpoint' explainer in the docs).[/dim]",
                title="[yellow]prepare required[/yellow]",
                border_style="yellow",
            )
        )
        return 2

    geom = _load_geometry(cfg)
    needed = None
    if geom is not None:
        book.show("vram_budget")
        needed = _print_vram_estimate(cfg, geom)
    else:
        console.print("[yellow]could not load model geometry — skipping the VRAM estimate[/yellow]")

    if needed is None:
        needed = 30.0
    preflight = preflight_gpus(cfg, needed)
    for warning in preflight.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    console.print(
        f"[bold]GPUs for this run:[/bold] {preflight.gpu_ids} "
        f"(free: {', '.join(f'{g}: {preflight.free_gb.get(g, float("nan")):.1f} GB' for g in preflight.gpu_ids)})"
    )

    if args.dry_run:
        console.print("[yellow]dry run: 3 steps, 8 examples, no eval — the full pipeline in miniature[/yellow]")

    run_dir = make_run_dir(cfg, dry_run=args.dry_run)
    cfg.snapshot(run_dir)
    write_env_fingerprint(run_dir, cfg, preflight.gpu_ids)
    console.print(f"[dim]run directory: {run_dir}[/dim]")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = _resolve_resume_checkpoint(args.resume)
        if resume_checkpoint is None:
            console.print(f"[red]no checkpoint found under {args.resume}[/red]")
            return 2
        console.print(
            f"[bold]resuming from[/bold] {resume_checkpoint}\n"
            "[dim]optimizer, scheduler and step counters are restored from the checkpoint; "
            "the new run dir records the resumed segment.[/dim]"
        )

    result = launch_worker(
        cfg,
        run_dir,
        preflight.gpu_ids,
        extra_args=(["--dry-run"] if args.dry_run else [])
        + (["--max-steps", str(args.max_steps)] if args.max_steps else [])
        + (["--resume", str(resume_checkpoint)] if resume_checkpoint else []),
    )
    if result.returncode != 0:
        console.print(
            Panel(
                f"training exited with code {result.returncode}\n"
                f"logs: {run_dir / 'logs'} · see docs/05-troubleshooting.md",
                title="[red]failure[/red]",
                border_style="red",
            )
        )
    return result.returncode


def cmd_report(args: argparse.Namespace) -> int:
    from .plots import replot, write_report

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")
    charts = replot(run_dir)
    report = write_report(run_dir)
    console.print(f"[green]✓[/green] regenerated {len(charts)} chart(s) in {run_dir / 'charts'}")
    console.print(f"[green]✓[/green] report: [bold]{report}[/bold]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .chat import run_chat

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
    from .model import merge_run

    merge_run(args.run, out_dir=args.out, console=console)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-finetune",
        description="Teaching-first QLoRA fine-tuning for AMD ROCm GPUs",
    )
    parser.add_argument("--version", action="version", version=f"llm-finetune {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=DEFAULT_CONFIG, help=f"TOML config (default: {DEFAULT_CONFIG})")
        p.add_argument(
            "--set", action="append", metavar="KEY=VALUE", help="arbitrary config override, e.g. --set optim.lr=2e-4"
        )
        p.add_argument("--quiet", action="store_true", help="disable explainer panels")

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
