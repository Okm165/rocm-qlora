"""SFTConfig assembly tests — the checkpoint cadence and dry-run logic (no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rocm_qlora.config import Config
from rocm_qlora.sft import build_sft_config

RUN_DIR = Path("/tmp/rocm-qlora-test-run")


def test_checkpoint_cadence_steps_vs_epoch():
    """`checkpoint_every_n_steps > 0` switches save cadence to steps (was: epoch-only)."""
    cfg = Config.from_dict({})
    cfg.logging.checkpoint_every_n_steps = 42
    sft = build_sft_config(cfg, RUN_DIR, dry_run=False, max_steps=None)
    assert str(sft.save_strategy).endswith("STEPS")
    assert sft.save_steps == 42

    cfg.logging.checkpoint_every_n_steps = 0
    sft = build_sft_config(cfg, RUN_DIR, dry_run=False, max_steps=None)
    assert str(sft.save_strategy).endswith("EPOCH")


def test_dry_run_is_tiny_and_saves_nothing():
    cfg = Config.from_dict({})
    sft = build_sft_config(cfg, RUN_DIR, dry_run=True, max_steps=None)
    assert sft.max_steps == 3
    assert sft.max_length <= 512
    assert str(sft.save_strategy).endswith("NO")
    assert str(sft.eval_strategy).endswith("NO")


def test_warmup_is_passed_as_the_ratio():
    """transformers v5 reads a `warmup_steps` float in [0, 1) as a ratio of the
    total optimizer steps (its own ceil arithmetic) — no schedule math here."""
    cfg = Config.from_dict({"optim": {"warmup_ratio": 0.03}})
    sft = build_sft_config(cfg, RUN_DIR, dry_run=False, max_steps=None)
    assert sft.warmup_steps == 0.03


def test_max_steps_override_wins():
    cfg = Config.from_dict({})
    sft = build_sft_config(cfg, RUN_DIR, dry_run=False, max_steps=50)
    assert sft.max_steps == 50


def test_gradient_checkpointing_kwargs_carry_sft_knobs():
    """offload / every_n_layers ride in the kwargs dict the trainer unpacks."""
    sft = build_sft_config(Config.from_dict({}), RUN_DIR, dry_run=False, max_steps=None)
    gc = sft.gradient_checkpointing_kwargs
    assert gc["use_reentrant"] is False
    assert gc["offload"] is False and gc["every_n_layers"] == 1

    cfg = Config.from_dict({"optim": {"activation_offloading": True, "checkpoint_every_n_layers": 2}})
    sft = build_sft_config(cfg, RUN_DIR, dry_run=False, max_steps=None)
    assert sft.gradient_checkpointing_kwargs["offload"] is True
    assert sft.gradient_checkpointing_kwargs["every_n_layers"] == 2


def test_dataloader_workers_are_a_plain_constant():
    """Two workers per rank: the dataset is tiny and pre-tokenized — no host
    core-count modelling."""
    sft = build_sft_config(Config.from_dict({}), RUN_DIR, dry_run=False, max_steps=None)
    assert sft.dataloader_num_workers == 2


def test_bundled_training_template_is_the_config_mapping():
    """The family is a config value; the table maps it to TRL's constants."""
    from trl.chat_template_utils import qwen3_8_training_chat_template, qwen3_training_chat_template

    from rocm_qlora.sft import _bundled_training_template

    assert _bundled_training_template("qwen3.8") == qwen3_8_training_chat_template
    assert _bundled_training_template("qwen3") == qwen3_training_chat_template
    assert _bundled_training_template("") is None  # no fallback declared
    assert _bundled_training_template("qwen4") is None  # (the Literal rejects this at load)


def test_training_template_fallback_installs_the_configured_family():
    """A modified community copy: TRL refuses, the configured family is installed."""
    from pathlib import Path

    from transformers import AutoTokenizer
    from trl.chat_template_utils import has_generation_markers

    from rocm_qlora.sft import ensure_training_chat_template

    prepared = Path(".data/models/qwen3.5-0.8b-nf4")
    if not prepared.exists():
        pytest.skip("toy prepared tokenizer not present")
    tokenizer = AutoTokenizer.from_pretrained(str(prepared), trust_remote_code=False)
    tokenizer.chat_template = str(tokenizer.chat_template) + "\n"  # modify the stock copy
    logs: list[str] = []

    cfg = Config.from_dict({"data": {"training_template": "qwen3.5_nothink"}})
    ensure_training_chat_template(tokenizer, cfg, log=logs.append)
    assert has_generation_markers(tokenizer.chat_template)
    assert logs and "qwen3.5_nothink" in logs[0]

    # no family declared: the run must abort rather than guess a template
    # (fresh copy of the modified stock template: the installed one now has markers)
    bare = AutoTokenizer.from_pretrained(str(prepared), trust_remote_code=False)
    bare.chat_template = str(bare.chat_template) + "\n"
    with pytest.raises(RuntimeError, match=r"data\.training_template"):
        ensure_training_chat_template(bare, Config.from_dict({}), log=logs.append)
