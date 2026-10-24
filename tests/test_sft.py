"""SFTConfig assembly tests — the checkpoint cadence and dry-run logic (no GPU)."""

from __future__ import annotations

from pathlib import Path

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


def test_bundled_family_template_matches_by_family_name():
    """The fallback for community checkpoints: the same template TRL would pick."""
    from trl.chat_template_utils import (
        qwen3_5_think_training_chat_template,
        qwen3_8_training_chat_template,
        qwen3_training_chat_template,
    )

    from rocm_qlora.sft import _bundled_family_template

    # specificity: a 3.8 name must not fall through to the generic qwen3 template
    assert _bundled_family_template("unsloth/Qwen3.8-27B-unsloth-bnb-4bit") == qwen3_8_training_chat_template
    assert _bundled_family_template("Qwen/Qwen3_5-0.8B") == qwen3_5_think_training_chat_template
    assert _bundled_family_template("Qwen/Qwen3-8B") == qwen3_training_chat_template
    # unknown family: None, so the caller raises instead of dropping the mask
    assert _bundled_family_template("meta-llama/Llama-3-8B") is None
