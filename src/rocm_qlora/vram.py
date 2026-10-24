"""VRAM estimation: parameter geometry -> per-GPU training footprint.

Pure math over parameter counts — no GPU, no rocm-smi, no torch. The inputs
come from the prepared checkpoint's ``finetune_meta.json`` (exact) or a
HuggingFace ``config.json`` (heuristic). The decomposition is deliberately
simple and explainable; the dashboard reports the real allocator numbers
during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict


@dataclass
class ModelGeometry:
    """Parameter accounting for the *prepared* (quantized) checkpoint.

    Numbers come from ``finetune_meta.json`` when the checkpoint was created
    by ``rocm-qlora prepare`` (exact), otherwise from ``config.json``
    (heuristic). Everything is in *parameters*, not bytes.
    """

    name: str = "unknown"
    total_params: float = 0.0
    embed_params: float = 0.0
    lm_head_params: float = 0.0  # 0 if tied
    quantized_params: float = 0.0
    nonquant_16bit_params: float = 0.0  # norms, convs, non-requantized extras
    vision_params: float = 0.0  # 0 if dropped in prepare
    # LoRA-relevant per-layer capacity (for trainable-parameter estimate):
    lora_units: float = 0.0  # sum over target linears of (in_features + out_features)
    hidden_size: int = 0
    n_layers: int = 0
    vocab_size: int = 0
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)


class QuantizationMeta(TypedDict):
    """The ``quantization`` section of finetune_meta.json."""

    quant_type: str
    double_quant: bool
    quant_storage: str
    compute_dtype: str


class FinetuneMeta(TypedDict):
    """`finetune_meta.json` in a prepared checkpoint: what `prepare` writes,
    what the VRAM estimator reads.

    The accounting keys are always written; the provenance extras may be
    missing in older checkpoints (readers stay tolerant via ``.get``).
    """

    name: str
    total_params: float
    embed_params: float
    lm_head_params: float
    quantized_params: float
    nonquant_16bit_params: float
    vision_params: float
    lora_units: float
    hidden_size: int
    n_layers: int
    vocab_size: int
    source: NotRequired[str]
    requantized: NotRequired[list[str]]
    created: NotRequired[str]
    quantization: NotRequired[QuantizationMeta]


def geometry_from_finetune_meta(meta_path: Path) -> ModelGeometry | None:
    """Geometry from the prepared checkpoint's finetune_meta.json."""
    try:
        meta: FinetuneMeta = json.loads(meta_path.read_text())
    except Exception:
        return None
    geom = ModelGeometry(
        name=meta.get("name", "unknown"),
        total_params=float(meta.get("total_params", 0.0)),
        embed_params=float(meta.get("embed_params", 0.0)),
        lm_head_params=float(meta.get("lm_head_params", 0.0)),
        quantized_params=float(meta.get("quantized_params", 0.0)),
        nonquant_16bit_params=float(meta.get("nonquant_16bit_params", 0.0)),
        vision_params=float(meta.get("vision_params", 0.0)),
        lora_units=float(meta.get("lora_units", 0.0)),
        hidden_size=int(meta.get("hidden_size", 0) or 0),
        n_layers=int(meta.get("n_layers", 0) or 0),
        vocab_size=int(meta.get("vocab_size", 0) or 0),
        source="finetune_meta",
    )
    geom.notes.append(f"geometry from {meta_path}")
    return geom


def geometry_from_hf_config(config_path: Path) -> ModelGeometry | None:
    """Heuristic geometry from a HF config.json (text config preferred)."""
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return None
    tc = cfg.get("text_config", cfg)
    hidden = tc.get("hidden_size")
    layers = tc.get("num_hidden_layers")
    vocab = tc.get("vocab_size")
    inter = tc.get("intermediate_size")
    if not all(isinstance(v, int) for v in (hidden, layers, vocab, inter)):
        return None
    tied = bool(tc.get("tie_word_embeddings", False))
    layer_types = tc.get("layer_types") or ["full_attention"] * layers
    n_linear = sum(1 for t in layer_types if "linear" in t)
    n_full = layers - n_linear

    mlp_per_layer = 3 * hidden * inter  # gate+up+down
    attn_per_layer, lora_units_attn = _attn_layer_sizes(tc, hidden)
    linear_per_layer, lora_units_linear, nonquant_linear = _linear_attn_layer_sizes(tc, hidden)
    lora_units_mlp = (hidden + inter) * 2 + (inter + hidden)

    vision_b = _vision_params_b(cfg)
    embed = vocab * hidden
    lm_head = 0.0 if tied else vocab * hidden
    core = n_full * (attn_per_layer + mlp_per_layer) + n_linear * (linear_per_layer + mlp_per_layer)
    total = embed + lm_head + core + vision_b * 1e9
    notes = ["geometry from config.json (heuristic)"]
    return ModelGeometry(
        name=str(cfg.get("_name_or_path", "unknown")),
        total_params=total,
        embed_params=embed,
        lm_head_params=lm_head,
        quantized_params=core * 0.9,  # most Linears quantized
        nonquant_16bit_params=core * 0.1 + n_linear * nonquant_linear,
        vision_params=vision_b * 1e9,
        lora_units=(n_full * (lora_units_attn + lora_units_mlp) + n_linear * (lora_units_linear + lora_units_mlp)),
        source="hf_config",
        notes=notes,
    )


def _attn_layer_sizes(tc: dict[str, Any], hidden: int) -> tuple[float, float]:
    """Per-layer attention params (q/k/v/o; Qwen3.5-style gated q_proj) and LoRA units."""
    heads = tc.get("num_attention_heads", 0)
    kv_heads = tc.get("num_key_value_heads", heads)
    head_dim = tc.get("head_dim", hidden // max(heads, 1))
    q_out = heads * head_dim * 2  # q_proj outputs query + gate
    per_layer = hidden * q_out + 2 * hidden * (kv_heads * head_dim) + (heads * head_dim) * hidden
    lora_units = (hidden + q_out) + 2 * (hidden + kv_heads * head_dim) + (heads * head_dim + hidden)
    return per_layer, lora_units


def _linear_attn_layer_sizes(tc: dict[str, Any], hidden: int) -> tuple[float, float, float]:
    """Gated DeltaNet per-layer params, LoRA units, and unquantized (16-bit) share.

    in_proj_qkv: hidden -> 2*k_dim + v_dim; in_proj_z: hidden -> v_dim;
    in_proj_a/b: hidden -> num_v_heads; out_proj: v_dim -> hidden.
    """
    lk_heads = tc.get("linear_num_key_heads", 0)
    lv_heads = tc.get("linear_num_value_heads", 0)
    lk_dim = tc.get("linear_key_head_dim", 0)
    lv_dim = tc.get("linear_value_head_dim", 0)
    if not (lk_heads and lv_heads and lk_dim and lv_dim):
        return 0.0, 0.0, 0.0
    k_dim, v_dim = lk_heads * lk_dim, lv_heads * lv_dim
    per_layer = (
        hidden * (2 * k_dim + v_dim)  # in_proj_qkv
        + hidden * v_dim  # in_proj_z
        + hidden * lv_heads  # in_proj_a
        + hidden * lv_heads  # in_proj_b
        + v_dim * hidden  # out_proj
    )
    lora_units = (hidden + 2 * k_dim + v_dim) + (hidden + v_dim) + (hidden + lv_heads) * 2 + (v_dim + hidden)
    nonquant_per_layer = hidden * (2 * (lk_heads * lk_dim) + lv_heads * lv_dim) * 0.9
    return per_layer, lora_units, nonquant_per_layer


def _vision_params_b(cfg: dict[str, Any]) -> float:
    """Rough vision-tower params in billions (attn+mlp ~4x expansion)."""
    if "vision_config" not in cfg or cfg.get("language_model_only", False):
        return 0.0
    vc = cfg["vision_config"]
    v_hidden = vc.get("hidden_size", 0)
    v_depth = vc.get("depth", 0)
    if not (v_hidden and v_depth):
        return 0.0
    return v_depth * 4 * v_hidden * v_hidden * 4 / 1e9


@dataclass
class VramEstimate:
    weights_gb: float
    lora_states_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float
    total_gb: float
    trainable_params: float
    notes: list[str] = field(default_factory=list)

    def breakdown(self) -> dict[str, float]:
        """Human-readable per-component VRAM breakdown lines."""
        return {
            "weights": self.weights_gb,
            "lora + optimizer": self.lora_states_gb,
            "activations": self.activations_gb,
            "logits spike": self.logits_gb,
            "CUDA/HIP context + misc": self.overhead_gb,
            "TOTAL": self.total_gb,
        }


def _weights_gb(geom: ModelGeometry, *, compute_dtype_bytes: float) -> float:
    """NF4 weights (~0.55 B/param) + the 16-bit parts (embeddings/head/vision)."""
    return (
        geom.quantized_params * 0.55
        + geom.nonquant_16bit_params * compute_dtype_bytes
        + geom.embed_params * compute_dtype_bytes
        + geom.lm_head_params * compute_dtype_bytes
        + geom.vision_params * compute_dtype_bytes
    ) / 1e9


def _lora_states_gb(
    geom: ModelGeometry, *, lora_r: int, optimizer: str, compute_dtype_bytes: float
) -> tuple[float, float]:
    """LoRA weights + grads + optimizer moments; returns (GB, trainable_params)."""
    trainable = geom.lora_units * lora_r
    lora_weights = trainable * compute_dtype_bytes
    lora_grads = trainable * compute_dtype_bytes
    if optimizer == "adamw_8bit":
        opt_bytes = 2 * trainable * 1.0  # two 8-bit moments (+ blockwise scales ~)
        opt_bytes *= 1.1
    elif optimizer == "adafactor":
        opt_bytes = trainable * 4 * 0.5  # factored second moments
    else:
        opt_bytes = trainable * 4 * 2  # two fp32 moments
    return (lora_weights + lora_grads + opt_bytes) / 1e9, trainable


def _activations_gb(geom: ModelGeometry, *, max_length: int, batch_size: int, grad_checkpointing: bool) -> float:
    """Stored activations (layer inputs only under checkpointing) + transient peaks.

    Stored activations scale with layers × batch × seq × hidden; gradient
    checkpointing recomputes them, keeping only layer inputs (~25%).
    """
    hidden = geom.hidden_size or 4096
    n_layers = geom.n_layers or max(1, int(geom.total_params / 4.3e8))
    per_layer_mb = batch_size * max_length * hidden * 2 / 1e6
    stored = n_layers * per_layer_mb * (0.25 if grad_checkpointing else 1.0)
    transient = per_layer_mb * 2.5  # recompute / MLP / attention peaks
    return (stored + transient) / 1024


def estimate_training_vram(
    geom: ModelGeometry,
    *,
    lora_r: int,
    max_length: int,
    batch_size: int,
    grad_checkpointing: bool = True,
    optimizer: str = "adamw_8bit",
    loss_type: str = "nll",
    compute_dtype_bytes: float = 2.0,
) -> VramEstimate:
    """Transparent VRAM estimate for one training process (one GPU).

    The formulas are deliberately simple and printed to the user; they are
    meant to be *directionally right and explainable*, while the dashboard
    reports the real numbers from the allocator during training.
    """
    weights = _weights_gb(geom, compute_dtype_bytes=compute_dtype_bytes)
    lora_states, trainable = _lora_states_gb(
        geom, lora_r=lora_r, optimizer=optimizer, compute_dtype_bytes=compute_dtype_bytes
    )
    activations = _activations_gb(
        geom, max_length=max_length, batch_size=batch_size, grad_checkpointing=grad_checkpointing
    )
    # TRL computes every SFT loss through a fused, chunked LM head, so full
    # fp32 logits are never materialized — `nll` and `dft` alike.
    logits = 0.0
    overhead = 1.0  # HIP context, RCCL buffers, allocator fragmentation margin
    total = weights + lora_states + activations + logits + overhead
    return VramEstimate(
        weights_gb=weights,
        lora_states_gb=lora_states,
        activations_gb=activations,
        logits_gb=logits,
        overhead_gb=overhead,
        total_gb=total,
        trainable_params=trainable,
        notes=[
            f"trainable ≈ {trainable / 1e6:.1f}M params (r={lora_r}, all-linear)",
            f"gradient checkpointing: {'on' if grad_checkpointing else 'off'}",
            f"loss: {loss_type}",
        ],
    )
