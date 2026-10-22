"""Config parsing/validation tests — the config is the user-facing contract."""

from __future__ import annotations

import pytest

from llm_finetune.config import Config, ConfigError, dumps_toml


def test_defaults_load():
    cfg = Config.from_dict({})
    assert cfg.lora.r == 16
    assert cfg.optim.loss_type == "nll"
    assert cfg.dist.mode == "auto"


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        Config.from_dict({"optim": {"learning_rate": 1e-4}})  # wrong key name


def test_partial_tables_merge_with_defaults():
    cfg = Config.from_dict({"optim": {"lr": 2e-4, "epochs": 2}})
    assert cfg.optim.lr == 2e-4
    assert cfg.optim.epochs == 2
    assert cfg.optim.grad_accum == 4  # untouched default


def test_validation_errors_are_collected():
    with pytest.raises(ConfigError) as exc:
        Config.from_dict({"optim": {"lr": -1}, "data": {"max_length": 4}})
    msg = str(exc.value)
    assert "optim.lr" in msg
    assert "data.max_length" in msg


def test_paged_optimizer_rejected_with_hint():
    with pytest.raises(ConfigError, match="paged optimizers are CUDA-only"):
        Config.from_dict({"optim": {"optimizer": "paged_adamw_8bit"}})


def test_overrides_apply_and_validate():
    cfg = Config.from_dict({})
    cfg2 = cfg.with_overrides({"optim.lr": 3e-4, "dist.gpus": [0, 2]})
    assert cfg2.optim.lr == 3e-4
    assert cfg2.dist.gpus == [0, 2]
    with pytest.raises(ConfigError):
        cfg.with_overrides({"optim.lr": 99})


def test_snapshot_roundtrip_toml(tmp_path):
    cfg = Config.from_dict({"run": {"name": "x"}, "data": {"max_length": 2048}})
    text = dumps_toml(cfg.to_dict())
    assert "[run]" in text and "[optim]" in text
    # a snapshot must be re-parseable
    (tmp_path / "c.toml").write_text(text)
    cfg2 = Config.from_toml(tmp_path / "c.toml")
    assert cfg2.data.max_length == 2048
    assert cfg2.run.name == "x"
