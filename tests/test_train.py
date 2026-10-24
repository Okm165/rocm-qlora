"""SFTConfig assembly tests — the checkpoint cadence and dry-run logic (no GPU)."""

from __future__ import annotations

from pathlib import Path

from rocm_qlora.config import Config
from rocm_qlora.train import build_sft_config

RUN_DIR = Path("/tmp/rocm-qlora-test-run")


def test_checkpoint_cadence_steps_vs_epoch():
    """`checkpoint_every_n_steps > 0` switches save cadence to steps (was: epoch-only)."""
    cfg = Config.from_dict({})
    cfg.logging.checkpoint_every_n_steps = 42
    sft = build_sft_config(cfg, RUN_DIR, world_size=3, n_train=999, dry_run=False, max_steps=None)
    assert str(sft.save_strategy).endswith("STEPS")
    assert sft.save_steps == 42

    cfg.logging.checkpoint_every_n_steps = 0
    sft = build_sft_config(cfg, RUN_DIR, world_size=3, n_train=999, dry_run=False, max_steps=None)
    assert str(sft.save_strategy).endswith("EPOCH")


def test_dry_run_is_tiny_and_saves_nothing():
    cfg = Config.from_dict({})
    sft = build_sft_config(cfg, RUN_DIR, world_size=1, n_train=999, dry_run=True, max_steps=None)
    assert sft.max_steps == 3
    assert sft.max_length <= 512
    assert str(sft.save_strategy).endswith("NO")
    assert str(sft.eval_strategy).endswith("NO")


def test_warmup_is_a_fraction_of_estimated_steps():
    """warmup_ratio is not a trainer arg in transformers v5 — we compute steps ourselves."""
    cfg = Config.from_dict({"optim": {"warmup_ratio": 0.03}})
    sft = build_sft_config(cfg, RUN_DIR, world_size=1, n_train=840, dry_run=False, max_steps=None)
    steps_per_epoch = -(-840 // (cfg.optim.per_device_batch_size * cfg.optim.grad_accum * 1))  # ceil
    expected = max(1, int(steps_per_epoch * cfg.optim.epochs * 0.03))
    assert sft.warmup_steps == expected


def test_max_steps_override_wins():
    cfg = Config.from_dict({})
    sft = build_sft_config(cfg, RUN_DIR, world_size=1, n_train=999, dry_run=False, max_steps=50)
    assert sft.max_steps == 50


def test_gradient_checkpointing_kwargs_carry_sft_knobs():
    """offload / every_n_layers ride in the kwargs dict the trainer unpacks."""
    sft = build_sft_config(Config.from_dict({}), RUN_DIR, world_size=3, n_train=999, dry_run=False, max_steps=None)
    gc = sft.gradient_checkpointing_kwargs
    assert gc["use_reentrant"] is False
    assert gc["offload"] is False and gc["every_n_layers"] == 1

    cfg = Config.from_dict({"optim": {"activation_offloading": True, "checkpoint_every_n_layers": 2}})
    sft = build_sft_config(cfg, RUN_DIR, world_size=3, n_train=999, dry_run=False, max_steps=None)
    assert sft.gradient_checkpointing_kwargs["offload"] is True
    assert sft.gradient_checkpointing_kwargs["every_n_layers"] == 2


def test_dataloader_worker_cap():
    """Two workers per rank is plenty for pre-tokenized data; fewer forks, less ROCm churn."""
    sft = build_sft_config(Config.from_dict({}), RUN_DIR, world_size=3, n_train=999, dry_run=False, max_steps=None)
    assert 1 <= sft.dataloader_num_workers <= 2
