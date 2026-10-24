"""Interactive chat: sanity-check a trained adapter.

This is the fastest way to answer "did it actually learn?" — load the base
model in 4-bit + the adapter, and talk to it. Differences from the base
model (style, reasoning structure, domain vocabulary) show up immediately.

Rendering: thinking blocks (` thinking…</think>`) are shown as dim text,
the final answer as Markdown — the same split the model was trained with.

Commands inside the loop:  /think on|off · /temp <x> · /max <n> · /reset · /exit
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from .config import Config
from .data import ChatMessage

if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


def _load(
    run_dir: Path | None, base_override: str | None, adapter_override: str | None, console: Console
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    cfg: Config | None = None
    if run_dir is not None:
        try:
            cfg = Config.from_toml(Path(run_dir) / "config.resolved.toml")
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
        attn_implementation=attn,
    )
    if adapter_dir and Path(adapter_dir).exists():
        console.print(f"[dim]attaching adapter {adapter_dir} …[/dim]")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return cast("PreTrainedModel", model), tokenizer


def _split_thinking(text: str) -> tuple[str | None, str]:
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        think = think.replace(" thinking", "").strip("\n ")
        return think.strip(), answer.strip()
    return None, text


def _print_banner(console: Console) -> None:
    """One-time command cheat-sheet line."""
    console.print("[dim]commands: /think on|off · /temp X · /max N · /reset · /exit[/dim]")


def _apply_command(
    user: str, gen: GenerationConfig, *, think: bool, history: list[ChatMessage], console: Console
) -> tuple[bool, bool]:
    """Handle a /command in place; returns ``(consumed, think)``."""
    if user == "/reset":
        history.clear()
        console.print("[dim]history cleared[/dim]")
        return True, think
    if user.startswith("/think"):
        think = user.split()[-1].lower() not in {"off", "false", "0"}
        console.print(f"[dim]thinking = {think}[/dim]")
        return True, think
    if user.startswith("/temp"):
        try:
            value = float(user.split()[-1])
            gen.temperature = max(value, 1e-5)
            gen.do_sample = value > 0
            console.print(f"[dim]temperature = {gen.temperature}[/dim]")
        except ValueError:
            console.print("[red]usage: /temp 0.7[/red]")
        return True, think
    if user.startswith("/max"):
        try:
            gen.max_new_tokens = int(user.split()[-1])
            console.print(f"[dim]max_new_tokens = {gen.max_new_tokens}[/dim]")
        except ValueError:
            console.print("[red]usage: /max 512[/red]")
        return True, think
    return False, think


def _build_prompt(tokenizer: PreTrainedTokenizerBase, history: list[ChatMessage], think: bool) -> str:
    """Chat-template the history, tolerating templates without thinking kwargs."""
    template_kwargs: dict[str, Any] = {"enable_thinking": think}
    try:
        rendered = tokenizer.apply_chat_template(  # TypedDict rows are plain dicts
            cast("list[dict[str, str]]", history), tokenize=False, add_generation_prompt=True, **template_kwargs
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            cast("list[dict[str, str]]", history), tokenize=False, add_generation_prompt=True
        )
    return cast("str", rendered)  # tokenize=False ⟹ str


def _generate_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    gen: GenerationConfig,
) -> tuple[str, int, float]:
    """One sampled generation; returns (text, new_tokens, seconds)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    generate: Any = model.generate  # GenerationMixin lives on the concrete model (stubs don't link it)
    with torch.inference_mode():
        out = generate(**inputs, generation_config=gen)
    dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = cast("str", tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
    return text, new_tokens, dt


def _render_reply(console: Console, text: str, new_tokens: int, dt: float) -> None:
    """Thinking block (when present) + Markdown answer + timing footer."""
    reasoning, answer = _split_thinking(text)
    if reasoning:
        console.print(reasoning, style="dim", markup=False)
    console.print(Markdown(answer or "(empty)"))
    console.print(f"[dim]{new_tokens} tokens · {new_tokens / dt:.1f} tok/s · {dt:.1f}s[/dim]")


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
    """Interactive loop: prompt → templated chat → sampled reply, with /commands."""
    console = console or Console()
    from transformers import GenerationConfig

    model, tokenizer = _load(Path(run_dir) if run_dir else None, base, adapter, console)
    gen = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-5),
        do_sample=temperature > 0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    think = True
    history: list[ChatMessage] = []
    _print_banner(console)

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
        consumed, think = _apply_command(user, gen, think=think, history=history, console=console)
        if consumed:
            continue

        history.append({"role": "user", "content": user})
        prompt = _build_prompt(tokenizer, history, think)
        text, new_tokens, dt = _generate_reply(model, tokenizer, prompt, gen)
        _render_reply(console, text, new_tokens, dt)
        history.append({"role": "assistant", "content": text})
