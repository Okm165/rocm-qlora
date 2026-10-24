"""Parameter accounting tests: the prepared-checkpoint meta and the model status report."""

from __future__ import annotations

import torch.nn as nn

from rocm_qlora.config import Config
from rocm_qlora.model import _prepare_meta, parameter_report


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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


def _model() -> _Fake27BShape:
    model = _Fake27BShape()
    model.config = type("C", (), {"hidden_size": 32, "num_hidden_layers": 3, "vocab_size": 64})()
    return model


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
