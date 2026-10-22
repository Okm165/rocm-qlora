"""Interactive chat: sanity-check a trained adapter.

This is the fastest way to answer "did it actually learn?" — load the base
model in 4-bit + the adapter, and talk to it. Differences from the base
model (style, reasoning structure, domain vocabulary) show up immediately.

Rendering: thinking blocks (` thinking…</think>`) are shown in a dim panel,
the final answer as Markdown — the same split the model was trained with.

Commands inside the loop:  /think on|off · /temp <x> · /max <n> · /reset · /exit
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Config

DEFAULT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": True}


def _load(run_dir: Path | None, base_override: str | None, adapter_override: str | None, console: Console):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    cfg: Config | None = None
    if run_dir is not None:
        try:
            cfg = Config.from_dict(json.loads((Path(run_dir) / "config.resolved.json").read_text()))
        except Exception:
            cfg = None

    base_dir = base_override or (cfg.model.prepared_dir if cfg else None)
    adapter_dir = adapter_override or (str(Path(run_dir) / "adapter") if run_dir else None)
    if base_dir is None:
        raise SystemExit("chat: provide --run <dir> or --base <prepared-model-dir>")

    tokenizer_src = adapter_dir if adapter_dir and Path(adapter_dir).exists() else base_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    from .launcher import resolve_attn_impl
    from .model import compute_dtype, resolve_model_class

    model_cls = resolve_model_class(str(base_dir))
    attn = resolve_attn_impl(cfg) if cfg else "sdpa"
    console.print(f"[dim]loading base {base_dir} (4-bit) …[/dim]")
    model = model_cls.from_pretrained(
        str(base_dir),
        dtype=compute_dtype(cfg) if cfg else torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    if adapter_dir and Path(adapter_dir).exists():
        console.print(f"[dim]attaching adapter {adapter_dir} …[/dim]")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def _split_thinking(text: str) -> tuple[str | None, str]:
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        think = think.replace(" thinking", "").strip("\n ")
        return think.strip(), answer.strip()
    return None, text


def run_chat(
    *,
    run_dir: str | None = None,
    base: str | None = None,
    adapter: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    console: Console | None = None,
) -> None:
    import torch

    console = console or Console()
    model, tokenizer = _load(Path(run_dir) if run_dir else None, base, adapter, console)
    think = True
    history: list[dict[str, str]] = []

    console.print(
        Panel(
            "Type a question. Commands: [bold]/think on|off[/bold] · [bold]/temp X[/bold] · "
            "[bold]/max N[/bold] · [bold]/reset[/bold] · [bold]/exit[/bold]",
            title="[bold]rocm-qlora chat[/bold]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user = Prompt.ask("[bold cyan]you[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        user = user.strip()
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history.clear()
            console.print("[dim]history cleared[/dim]")
            continue
        if user.startswith("/think"):
            think = user.split()[-1].lower() not in {"off", "false", "0"}
            console.print(f"[dim]thinking = {think}[/dim]")
            continue
        if user.startswith("/temp"):
            try:
                temperature = float(user.split()[-1])
                console.print(f"[dim]temperature = {temperature}[/dim]")
            except ValueError:
                console.print("[red]usage: /temp 0.7[/red]")
            continue
        if user.startswith("/max"):
            try:
                max_new_tokens = int(user.split()[-1])
                console.print(f"[dim]max_new_tokens = {max_new_tokens}[/dim]")
            except ValueError:
                console.print("[red]usage: /max 512[/red]")
            continue

        history.append({"role": "user", "content": user})
        template_kwargs = {"enable_thinking": think} if think is not None else {}
        try:
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.perf_counter() - t0
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        reasoning, answer = _split_thinking(text)
        if reasoning:
            console.print(Panel(reasoning, title="[dim]thinking[/dim]", border_style="grey37", padding=(0, 1)))
        console.print(
            Panel(Markdown(answer or "(empty)"), title="[bold green]assistant[/bold green]", border_style="green")
        )
        console.print(f"[dim]{new_tokens} tokens · {new_tokens / dt:.1f} tok/s · {dt:.1f}s[/dim]")
        history.append({"role": "assistant", "content": text})
