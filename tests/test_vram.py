"""VRAM estimator + model geometry tests (pure math, no GPU needed)."""

from __future__ import annotations

import json

from rocm_qlora.vram import (
    ModelGeometry,
    estimate_training_vram,
    geometry_from_finetune_meta,
    geometry_from_hf_config,
)


def make_geom():
    return ModelGeometry(
        name="test",
        total_params=27e9,
        embed_params=1.27e9,
        lm_head_params=1.27e9,
        quantized_params=22e9,
        nonquant_16bit_params=0.2e9,
        vision_params=0.5e9,
        lora_units=7e6,
        hidden_size=5120,
        n_layers=64,
        vocab_size=248320,
    )


def test_vram_estimate_monotonicity():
    geom = make_geom()
    base = estimate_training_vram(geom, lora_r=16, max_length=1024, batch_size=1)
    longer = estimate_training_vram(geom, lora_r=16, max_length=2048, batch_size=1)
    bigger_batch = estimate_training_vram(geom, lora_r=16, max_length=1024, batch_size=2)
    bigger_rank = estimate_training_vram(geom, lora_r=64, max_length=1024, batch_size=1)
    assert longer.total_gb > base.total_gb
    assert bigger_batch.total_gb > base.total_gb
    assert bigger_rank.total_gb > base.total_gb
    # TRL's fused chunked head never materializes full logits — any loss type
    assert base.logits_gb == 0.0
    nll = estimate_training_vram(geom, lora_r=16, max_length=1024, batch_size=1, loss_type="nll")
    assert nll.logits_gb == 0.0


def test_geometry_from_meta(tmp_path):
    meta = tmp_path / "finetune_meta.json"
    meta.write_text(json.dumps({"total_params": 1e9, "hidden_size": 2048, "n_layers": 24, "lora_units": 1e6}))
    geom = geometry_from_finetune_meta(meta)
    assert geom is not None and geom.n_layers == 24 and geom.source == "finetune_meta"


def test_geometry_from_hf_config_hybrid_with_vision(tmp_path):
    cfg = {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 152064,
            "intermediate_size": 13824,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": False,
            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
        "vision_config": {"hidden_size": 1152, "depth": 27},
        "_name_or_path": "fake/Qwen3.8-27B",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    geom = geometry_from_hf_config(path)
    assert geom is not None
    # geometry_from_hf_config fills parameter counts only; hidden_size/n_layers
    # stay 0 and the VRAM estimator falls back (see _activations_gb).
    assert geom.n_layers == 0 and geom.hidden_size == 0
    assert geom.quantized_params > geom.nonquant_16bit_params
    assert geom.lm_head_params > 0  # untied embeddings
    assert geom.vision_params > 0
    assert geom.total_params > 20e9


def test_geometry_from_hf_config_plain_tied(tmp_path):
    cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 152064,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    geom = geometry_from_hf_config(path)
    assert geom is not None
    assert geom.lm_head_params == 0.0  # tied
    assert geom.vision_params == 0.0
    assert geom.total_params > 0


def test_geometry_from_hf_config_rejects_missing_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hidden_size": 4096}))  # missing layers/vocab/inter
    assert geometry_from_hf_config(path) is None
