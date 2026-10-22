"""Dataset loading/validation/profile tests (no model downloads — stub tokenizer)."""

from __future__ import annotations

import json

import pytest

from llm_finetune.data import (
    DatasetError,
    load_sharegpt,
    profile_tokens,
    render_preview,
    split_by_index,
)


class StubTokenizer:
    """~1 token per 4 characters; good enough for length logic."""

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return list(range(len(text) // 4)) if tokenize else text


def write_dataset(tmp_path, examples):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(examples))
    return p


def make_example(q_len=40, a_len=200, think=False):
    answer = "x" * a_len
    if think:
        # tags built programmatically: angle-bracket literals can be mangled
        # by editors/sanitizers, and this fixture must be exact.
        open_t, close_t = "<" + "think" + ">", "</" + "think" + ">"
        answer = open_t + "reasoning" + close_t + answer
    return {"messages": [{"role": "user", "content": "q" * q_len}, {"role": "assistant", "content": answer}]}


def test_load_and_stats(tmp_path):
    p = write_dataset(tmp_path, [make_example(think=True), make_example()])
    examples, stats = load_sharegpt(p)
    assert len(examples) == 2
    assert stats.n_examples == 2
    assert stats.think_blocks == 1
    assert stats.think_complete == 1
    assert stats.turns_per_example == {2: 2}


def test_invalid_role_rejected(tmp_path):
    bad = {"messages": [{"role": "user", "content": "q"}, {"role": "robot", "content": "a"}]}
    p = write_dataset(tmp_path, [bad])
    with pytest.raises(DatasetError, match="invalid role"):
        load_sharegpt(p)


def test_empty_content_rejected(tmp_path):
    bad = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]}
    p = write_dataset(tmp_path, [bad])
    with pytest.raises(DatasetError, match="empty content"):
        load_sharegpt(p)


def test_missing_file_error_is_actionable(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_sharegpt(tmp_path / "nope.json")


def test_profile_and_coverage():
    examples = [make_example(a_len=400)] * 10
    tok = StubTokenizer()
    profile = profile_tokens(tok, examples, max_length=300)
    assert profile.n_total == 10
    assert profile.total_tokens > 0
    assert 0.0 <= profile.coverage_at(300) <= 1.0
    # coverage must be monotonic in the cap
    assert profile.coverage_at(1000) >= profile.coverage_at(300)


def test_render_preview_uses_template():
    tok = StubTokenizer()
    text = render_preview(tok, make_example())
    assert "<|user|>" in text and "<|assistant|>" in text


def test_split_is_deterministic_and_disjoint():
    train1, eval1 = split_by_index(100, eval_fraction=0.1, seed=42)
    train2, eval2 = split_by_index(100, eval_fraction=0.1, seed=42)
    assert train1 == train2 and eval1 == eval2
    assert set(train1).isdisjoint(eval1)
    assert len(eval1) == 10
