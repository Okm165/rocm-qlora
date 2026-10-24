"""Parameter accounting tests — the atomic classification the VRAM estimator trusts."""

from __future__ import annotations

import torch.nn as nn

from rocm_qlora.config import Config
from rocm_qlora.model import model_parameter_meta


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


def test_model_parameter_meta_buckets():
    """embed > lm_head > vision > quantized > nonquant, plus lora_units accounting."""
    model = _model()
    meta = model_parameter_meta(model, Config.from_dict({}))

    assert meta["embed_params"] == 64 * 32
    assert meta["lm_head_params"] == 64 * 32
    assert meta["vision_params"] == 32 * 32 + 32  # weight + bias
    assert meta["quantized_params"] == 0.0  # no bnb layers on this CPU-only stand-in
    assert meta["total_params"] == float(sum(p.numel() for p in model.parameters()))
    # 3 blocks x {q,v,o}_proj x (in_features + out_features)
    assert meta["lora_units"] == 3 * 3 * (32 + 32)
    assert meta["hidden_size"] == 32 and meta["n_layers"] == 3 and meta["vocab_size"] == 64
