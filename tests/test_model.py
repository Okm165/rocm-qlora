"""Parameter accounting tests: the prepared-checkpoint meta and the model status report."""

from __future__ import annotations

import torch.nn as nn

from rocm_qlora.config import Config
from rocm_qlora.model import parameter_report
from rocm_qlora.prepare import _prepare_meta


class _Fake27BShape(nn.Module):
    """Tiny stand-in for the real geometry: embed/lm_head/vision/attention blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 32)
        self.lm_head = nn.Linear(32, 64, bias=False)
        self.visual = nn.Linear(32, 32)
        self.layers = nn.ModuleList()
        for _ in range(3):
            block = nn.Module()
            block.q_proj = nn.Linear(32, 32, bias=False)
            block.v_proj = nn.Linear(32, 32, bias=False)
            block.o_proj = nn.Linear(32, 32, bias=False)
            self.layers.append(block)


def _model() -> _Fake27BShape:
    return _Fake27BShape()


def test_prepare_meta_counts_tied_weights_once():
    """A tied embed/lm_head is one tensor; the meta must count it once."""
    model = _model()
    model.lm_head.weight = model.embed_tokens.weight
    cfg = Config.from_dict({})
    meta = _prepare_meta(model, cfg, requantized=["layers.0.q_proj"])

    assert meta["total_params"] == float(sum(p.numel() for p in model.parameters()))
    assert meta["source"] == cfg.model.name
    assert meta["requantized"] == ["layers.0.q_proj"]
    # the quantization settings live in the checkpoint's own config.json, not here
    assert "quantization" not in meta


def test_skip_list_uses_full_paths_not_suffixes():
    """A 16-bit vision `qkv` must not skip the quantized `in_proj_qkv` at load."""
    from types import SimpleNamespace

    import torch.nn as nn

    from rocm_qlora.prepare import _write_quantization_config

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Module()
            self.visual.qkv = nn.Linear(4, 4, bias=False)  # plain (16-bit) Linear
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(4, 4, bias=False)  # plain here too
            self.config = SimpleNamespace()

    model = Fake()
    _write_quantization_config(model, Config.from_dict({}))
    skip = model.config.quantization_config["llm_int8_skip_modules"]
    assert skip == ["linear_attn.in_proj_qkv", "visual.qkv"]

    from transformers.quantizers.quantizers_utils import should_convert_module

    # a wrapped name (PEFT prefixes it) still matches by suffix
    assert should_convert_module("base_model.model.visual.qkv", skip) is False
    # and nothing else is caught by a short suffix
    assert should_convert_module("language_model.layers.0.linear_attn.in_proj_z", skip) is True


def test_parameter_report_buckets_by_peft_module_name():
    """PEFT's `...q_proj.lora_A.weight` buckets under q_proj, not under lora_A."""
    model = _model()
    for param in model.parameters():
        param.requires_grad_(False)
    model.layers[0].q_proj.lora_A = nn.Linear(32, 4, bias=False)  # type: ignore[attr-defined]
    model.layers[0].q_proj.lora_B = nn.Linear(4, 32, bias=False)  # type: ignore[attr-defined]

    report = parameter_report(model)
    assert report["trainable"] == 32 * 4 + 4 * 32
    assert report["by_module"] == {"q_proj": 32 * 4 + 4 * 32}


def test_logical_numel_uses_the_quant_state_shape():
    """Packed 4-bit storage under-reports `numel()`; quant_state has the logical shape."""
    from types import SimpleNamespace

    import bitsandbytes as bnb
    import torch

    from rocm_qlora.model import logical_numel

    packed = bnb.nn.Params4bit(torch.zeros(64, 1, dtype=torch.uint8))  # 64 packed bytes
    packed.quant_state = SimpleNamespace(shape=(8, 16))  # two 4-bit values per byte
    assert logical_numel(packed) == 128
    assert logical_numel(nn.Parameter(torch.zeros(3, 4))) == 12


def test_prepared_checkpoint_requires_the_finish_marker(tmp_path):
    """`finetune_meta.json` is written last — a config-only dir is a crashed save."""
    from rocm_qlora.model import prepared_checkpoint_exists

    cfg = Config.from_dict({"model": {"prepared_dir": str(tmp_path)}})
    assert prepared_checkpoint_exists(cfg) is False
    (tmp_path / "config.json").write_text("{}")
    assert prepared_checkpoint_exists(cfg) is False
    (tmp_path / "finetune_meta.json").write_text("{}")
    assert prepared_checkpoint_exists(cfg) is True
