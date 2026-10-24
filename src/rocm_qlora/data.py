"""Dataset handling: load, validate, split, profile.

The dataset format is **ShareGPT-style**: a JSON list of objects with a
``messages`` array of ``{"role": ..., "content": ...}`` turns. This module:

* loads and *validates* the file with actionable error messages (a broken
  dataset should fail loudly before the GPU spins up),
* computes character- and token-level statistics used to choose
  ``max_length`` (coverage at the cap, truncation risk),
* produces the HuggingFace ``Dataset``/``DatasetDict`` objects TRL consumes,
* renders previews with the model's real chat template so the exact
  rendering the model will be trained on is visible.

Tokenizer-agnostic by design: anything with ``apply_chat_template`` works,
which keeps unit tests fast (stub tokenizer, no downloads).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from .utils import percentile

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

VALID_ROLES = {"system", "user", "assistant", "tool"}


class ChatMessage(TypedDict):
    """One validated conversation turn."""

    role: str
    content: str


class DataProfile(TypedDict):
    """The ``profile.json`` artifact written by the ``rocm-qlora data`` command."""

    n_examples: int
    percentiles: dict[str, float]
    coverage_at_max_length: float
    max_length: int
    truncated: int
    total_tokens: int
    suggested_max_length_p99: int
    think_blocks: int


@dataclass(frozen=True)
class Percentiles:
    """The five-point length profile used to choose ``max_length``."""

    p50: float
    p90: float
    p95: float
    p99: float
    p100: float


def _percentiles(values: Sequence[float]) -> Percentiles:
    p50, p90, p95, p99, p100 = (percentile(values, q) for q in (50, 90, 95, 99, 100))
    return Percentiles(p50, p90, p95, p99, p100)


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """User-facing dataset problem (file layout, roles, empty content)."""


@dataclass
class DatasetStats:
    n_examples: int
    role_counts: dict[str, int]
    turns_per_example: dict[int, int]
    assistant_chars: list[int]
    user_chars: list[int]
    think_blocks: int  # assistant turns containing  thinking...</think>
    think_complete: int  # ...with a matching closing tag

    def char_percentiles(self, which: str = "assistant") -> Percentiles:
        """Character-length percentiles of assistant (or user) turns."""
        return _percentiles(self.assistant_chars if which == "assistant" else self.user_chars)


def load_sharegpt(
    path: str | Path, messages_field: str = "messages"
) -> tuple[list[dict[str, list[ChatMessage]]], DatasetStats]:
    """Load + validate a ShareGPT-style JSON file."""
    path = Path(path)
    raw = _read_json_list(path)

    examples: list[dict[str, list[ChatMessage]]] = []
    role_counts: dict[str, int] = {}
    turns_hist: dict[int, int] = {}
    assistant_chars: list[int] = []
    user_chars: list[int] = []
    think_blocks = think_complete = 0

    for i, item in enumerate(raw):
        messages = _validated_messages(path, i, item, messages_field)
        roles_here: set[str] = set()
        for msg in messages:
            role, content = msg["role"], msg["content"]
            role_counts[role] = role_counts.get(role, 0) + 1
            roles_here.add(role)
            if role == "assistant":
                assistant_chars.append(len(content))
                if "<think" in content:
                    think_blocks += 1
                    if "</think" in content:
                        think_complete += 1
            elif role == "user":
                user_chars.append(len(content))
        if "assistant" not in roles_here or "user" not in roles_here:
            raise DatasetError(f"{path}: example {i} must contain both user and assistant turns")
        turns_hist[len(messages)] = turns_hist.get(len(messages), 0) + 1
        examples.append({messages_field: messages})

    stats = DatasetStats(
        n_examples=len(examples),
        role_counts=role_counts,
        turns_per_example=dict(sorted(turns_hist.items())),
        assistant_chars=sorted(assistant_chars),
        user_chars=sorted(user_chars),
        think_blocks=think_blocks,
        think_complete=think_complete,
    )
    return examples, stats


def _read_json_list(path: Path) -> list[Any]:
    """Read `path` as JSON, raising DatasetError for a missing file or non-list."""
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}\nIf the raw file was moved, update data.path in the config.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise DatasetError(f"{path}: expected a non-empty JSON list of examples")
    return raw


def _validated_messages(path: Path, i: int, item: Any, messages_field: str) -> list[ChatMessage]:
    """Validate example `i` and every turn; returns the type-checked turns."""
    if not isinstance(item, dict) or messages_field not in item:
        raise DatasetError(f"{path}: example {i} has no '{messages_field}' key")
    messages = item[messages_field]
    if not isinstance(messages, list) or len(messages) < 2:
        raise DatasetError(f"{path}: example {i}: '{messages_field}' must have >= 2 turns")
    for j, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise DatasetError(
                f"{path}: example {i}, turn {j}: invalid role {role!r} (expected one of {sorted(VALID_ROLES)})"
            )
        if not isinstance(content, str) or not content.strip():
            raise DatasetError(f"{path}: example {i}, turn {j}: empty content")
    return cast("list[ChatMessage]", messages)


# ---------------------------------------------------------------------------
# Token-level profiling
# ---------------------------------------------------------------------------


@dataclass
class TokenProfile:
    lengths: list[int]  # tokens per example (sorted)
    max_length: int
    n_truncated: int
    n_total: int
    total_tokens: int

    @property
    def truncation_pct(self) -> float:
        """Percentage of examples truncated at max_length."""
        return 100.0 * self.n_truncated / max(1, self.n_total)

    def percentiles(self) -> Percentiles:
        """Token-length percentiles (p50/p90/p95/p99/p100)."""
        return _percentiles(self.lengths)

    def coverage_at(self, cap: int) -> float:
        """Fraction of examples that fit under ``cap`` tokens."""
        if not self.lengths:
            return float("nan")
        return sum(1 for length in self.lengths if length <= cap) / len(self.lengths)


def profile_tokens(
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict[str, list[ChatMessage]]],
    *,
    messages_field: str = "messages",
    max_length: int = 1024,
    template_kwargs: dict[str, Any] | None = None,
) -> TokenProfile:
    """Tokenize every example with the model's chat template and measure.

    Note: TRL tokenizes with its own (possibly patched) template during
    training; the numbers here are the right ballpark for choosing
    ``max_length``. The profiler measures the rendered conversation *including* the
    assistant turn (i.e., exactly what the model is trained on).
    """
    lengths: list[int] = []
    for item in examples:
        try:
            ids: Any = tokenizer.apply_chat_template(  # dynamic union; normalized just below
                cast("list[dict[str, str]]", item[messages_field]),  # TypedDict rows are plain dicts
                tokenize=True,
                add_generation_prompt=False,
                **(template_kwargs or {}),
            )
            # transformers v5 may return a BatchEncoding/dict, a tensor, or a list.
            if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            while ids and isinstance(ids[0], list):  # strip batch dims
                ids = ids[0]
            lengths.append(len(ids))
        except Exception:
            # Fall back to a character heuristic (~3.6 chars/token) rather
            # than failing the whole profile.
            text = "".join(m["content"] for m in item[messages_field])
            lengths.append(int(len(text) / 3.6))
    lengths.sort()
    return TokenProfile(
        lengths=lengths,
        max_length=max_length,
        n_truncated=sum(1 for length in lengths if length > max_length),
        n_total=len(lengths),
        total_tokens=sum(lengths),
    )


def render_preview(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, list[ChatMessage]],
    *,
    messages_field: str = "messages",
    template_kwargs: dict[str, Any] | None = None,
    max_chars: int = 1400,
) -> str:
    """Render one example with the real chat template (for teaching output)."""
    try:
        text = cast(
            "str",
            tokenizer.apply_chat_template(
                cast("list[dict[str, str]]", example[messages_field]),  # TypedDict rows are plain dicts
                tokenize=False,
                add_generation_prompt=False,
                **(template_kwargs or {}),
            ),
        )  # tokenize=False ⟹ str
    except Exception as exc:  # pragma: no cover - depends on tokenizer quirks
        return f"<chat template failed: {exc}>"
    text = text.replace("<|im_start|>", "⟨im_start⟩").replace("<|im_end|>", "⟨im_end|⟩")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… [truncated for display, {len(text):,} chars total]"
    return text


# ---------------------------------------------------------------------------
# HuggingFace Dataset conversion
# ---------------------------------------------------------------------------


def to_hf_dataset(examples: list[dict[str, list[ChatMessage]]], messages_field: str = "messages"):
    """Wrap examples as a HF Dataset with only the messages column."""
    from datasets import Dataset

    return Dataset.from_list([{messages_field: item[messages_field]} for item in examples])


def split_dataset(ds: Dataset, *, eval_fraction: float, seed: int):
    """Train/eval split with a fixed seed. Returns DatasetDict."""
    if eval_fraction <= 0:
        from datasets import DatasetDict

        return DatasetDict({"train": ds})
    return ds.train_test_split(test_size=eval_fraction, seed=seed, shuffle=True)
