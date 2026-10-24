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

#: The dataset's reasoning-trace delimiters (docs/01). Built from pieces:
#: angle brackets written as one literal have been mangled by tooling before.
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"


class ChatMessage(TypedDict):
    """One validated conversation turn."""

    role: str
    content: str


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
    think_blocks: int  # assistant turns containing THINK_OPEN
    think_complete: int  # ...with a matching closing tag

    def char_percentiles(self) -> Percentiles:
        """Character-length percentiles of the assistant turns."""
        return _percentiles(self.assistant_chars)


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
                # docs/01's dataset convention: THINK_OPEN ... THINK_CLOSE traces
                opened_at = content.find(THINK_OPEN)
                if opened_at != -1:
                    think_blocks += 1
                    if content.find(THINK_CLOSE) > opened_at:
                        think_complete += 1
        if "assistant" not in roles_here or "user" not in roles_here:
            raise DatasetError(f"{path}: example {i} must contain both user and assistant turns")
        turns_hist[len(messages)] = turns_hist.get(len(messages), 0) + 1
        examples.append({messages_field: messages})

    stats = DatasetStats(
        n_examples=len(examples),
        role_counts=role_counts,
        turns_per_example=dict(sorted(turns_hist.items())),
        assistant_chars=sorted(assistant_chars),
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
        if not isinstance(msg, dict):
            raise DatasetError(f"{path}: example {i}, turn {j} is not an object")
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise DatasetError(
                f"{path}: example {i}, turn {j}: invalid role {role!r} (expected one of {sorted(VALID_ROLES)})"
            )
        if not isinstance(content, str):
            raise DatasetError(f"{path}: example {i}, turn {j}: content must be a string")
        if not content.strip():
            raise DatasetError(f"{path}: example {i}, turn {j}: empty content")
    return cast("list[ChatMessage]", messages)


# ---------------------------------------------------------------------------
# Token-level profiling
# ---------------------------------------------------------------------------


@dataclass
class TokenProfile:
    lengths: list[int]  # tokens per example (sorted)
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


def _token_count(tokenizer: PreTrainedTokenizerBase, messages: list[ChatMessage]) -> int:
    """Chat-templated token count for one example.

    Template failures propagate (``profile_tokens`` names the example): the
    same template renders the training data, so an estimate here would poison
    the length profile that `max_length` is chosen from.
    """
    ids = tokenizer.apply_chat_template(
        cast("list[dict[str, str]]", messages),  # TypedDict rows are plain dicts
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,  # a plain list[int], not a BatchEncoding
    )
    return len(cast("list[int]", ids))


def profile_tokens(
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict[str, list[ChatMessage]]],
    *,
    messages_field: str = "messages",
    max_length: int = 1024,
) -> TokenProfile:
    """Tokenize every example with the tokenizer's current chat template and measure.

    Both callers install the *training* template first (``data`` and the
    training worker), so these are the token counts training actually renders —
    including the assistant turn, which is what the loss is computed on.
    """
    lengths: list[int] = []
    for index, item in enumerate(examples):
        try:
            lengths.append(_token_count(tokenizer, item[messages_field]))
        except Exception as exc:
            raise DatasetError(f"chat template failed on example {index}: {exc}") from exc
    lengths.sort()
    return TokenProfile(
        lengths=lengths,
        n_truncated=sum(1 for length in lengths if length > max_length),
        n_total=len(lengths),
        total_tokens=sum(lengths),
    )


def render_preview(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, list[ChatMessage]],
    *,
    messages_field: str = "messages",
    max_chars: int = 1400,
) -> str:
    """Render one example with the real chat template (shown by the `data` command)."""
    try:
        text = cast(
            "str",
            tokenizer.apply_chat_template(
                cast("list[dict[str, str]]", example[messages_field]),  # TypedDict rows are plain dicts
                tokenize=False,
                add_generation_prompt=False,
            ),
        )  # tokenize=False ⟹ str
    except Exception as exc:  # pragma: no cover - depends on tokenizer quirks
        return f"<chat template failed: {exc}>"
    text = text.replace("<|im_start|>", "⟨im_start⟩").replace("<|im_end|>", "⟨im_end⟩")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… [truncated for display, {len(text):,} chars total]"
    return text


# ---------------------------------------------------------------------------
# HuggingFace Dataset conversion
# ---------------------------------------------------------------------------


def to_hf_dataset(examples: list[dict[str, list[ChatMessage]]], messages_field: str = "messages"):
    """Wrap examples as a HF Dataset, always under TRL's `messages` key.

    The column name is not configurable: `messages` is the key TRL's SFTTrainer
    looks for in conversational data (and what `assistant_only_loss` requires).
    ``data.messages_field`` only names the key inside the raw JSON file.
    """
    from datasets import Dataset

    return Dataset.from_list([{"messages": item[messages_field]} for item in examples])


def split_dataset(ds: Dataset, *, eval_fraction: float, seed: int):
    """Train/eval split with a fixed seed. Returns DatasetDict."""
    if eval_fraction <= 0:
        from datasets import DatasetDict

        return DatasetDict({"train": ds})
    return ds.train_test_split(test_size=eval_fraction, seed=seed, shuffle=True)
