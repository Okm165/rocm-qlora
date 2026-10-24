"""Dataset loading/validation/profile tests (no model downloads — stub tokenizer)."""

from __future__ import annotations

import json

import pytest

from rocm_qlora.data import (
    THINK_CLOSE,
    THINK_OPEN,
    DatasetError,
    load_sharegpt,
    profile_tokens,
    render_preview,
    split_dataset,
    to_hf_dataset,
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


def test_to_hf_dataset_and_split_are_seeded():
    examples = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(20)
    ]
    ds = to_hf_dataset(examples, "messages")
    assert len(ds) == 20
    split_a = split_dataset(ds, eval_fraction=0.2, seed=42)
    split_b = split_dataset(ds, eval_fraction=0.2, seed=42)
    assert len(split_a["train"]) == 16 and len(split_a["test"]) == 4
    assert split_a["train"]["messages"] == split_b["train"]["messages"]  # seeded → deterministic


def test_non_dict_turn_is_a_dataset_error(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([{"messages": [{"role": "user", "content": "hi"}, "oops"]}]))
    with pytest.raises(DatasetError, match="turn 1 is not an object"):
        load_sharegpt(path)


def test_hf_dataset_always_uses_trl_messages_key(tmp_path):
    """`data.messages_field` names the raw file's key; TRL only understands `messages`."""
    path = tmp_path / "ds.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
                {"conversations": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "later"}]},
            ]
        )
    )
    examples, _ = load_sharegpt(path, messages_field="conversations")
    ds = to_hf_dataset(examples, messages_field="conversations")
    assert ds.column_names == ["messages"]


def test_think_stats_use_the_documented_tags_and_order(tmp_path):
    """` thinking` opens, `</think>` closes — a close before the open is not complete."""
    path = tmp_path / "ds.json"
    path.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": THINK_OPEN + "here" + THINK_CLOSE + "answer"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": THINK_CLOSE + "stray" + THINK_CLOSE + "no open tag"},
                    ]
                },
            ]
        )
    )
    _, stats = load_sharegpt(path)
    assert stats.think_blocks == 1 and stats.think_complete == 1
