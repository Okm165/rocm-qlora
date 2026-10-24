"""Chat pure helpers (no model load) — the /command surface and thinking split."""

from __future__ import annotations

from rich.console import Console
from transformers import GenerationConfig

from rocm_qlora.chat import _apply_command, _split_thinking


def test_split_thinking():
    think, answer = _split_thinking("reasoning here</think>The answer.")
    assert think == "reasoning here" and answer == "The answer."
    assert _split_thinking("plain answer") == (None, "plain answer")


def test_apply_command_mutates_generation_config():
    console = Console()
    gen = GenerationConfig(max_new_tokens=64, temperature=0.6, do_sample=True)
    history = [{"role": "user", "content": "hi"}]

    consumed, think = _apply_command("/temp 0.2", gen, think=True, history=history, console=console)
    assert consumed and think is True
    assert abs(gen.temperature - 0.2) < 1e-9 and gen.do_sample is True

    _, _ = _apply_command("/temp 0", gen, think=True, history=history, console=console)
    assert gen.do_sample is False and abs(gen.temperature - 0.2) < 1e-9  # greedy; temperature unused

    _, _ = _apply_command("/max 512", gen, think=True, history=history, console=console)
    assert gen.max_new_tokens == 512

    consumed, think = _apply_command("/think off", gen, think=True, history=history, console=console)
    assert consumed and think is False

    consumed, _ = _apply_command("/reset", gen, think=False, history=history, console=console)
    assert consumed and history == []

    assert _apply_command("hello", gen, think=True, history=history, console=console) == (False, True)
