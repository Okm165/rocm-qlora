"""Merge sizing: exact bf16 bytes from the headers, device choice from facts."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

from rich.console import Console

from rocm_qlora.merge import _bf16_model_bytes, _choose_merge_device


def write_fake_checkpoint(tmp_path: Path) -> Path:
    """A safetensors file with headers only — `_bf16_model_bytes` reads no data."""
    header = {
        "layer.weight": {"dtype": "U8", "shape": [128], "data_offsets": [0, 128]},
        "layer.weight.absmax": {"dtype": "U8", "shape": [2], "data_offsets": [128, 130]},
        "layer.weight.quant_state.bitsandbytes__nf4": {"dtype": "U8", "shape": [16], "data_offsets": [130, 146]},
        "embed_tokens.weight": {"dtype": "F16", "shape": [100, 2], "data_offsets": [146, 546]},
        "lm_head.weight": {"dtype": "BF16", "shape": [50], "data_offsets": [546, 646]},
        "__metadata__": {"format": "pt"},
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return tmp_path


def test_bf16_model_bytes_counts_parameters_not_disk_bytes(tmp_path):
    # 128 U8 bytes pack 256 NF4 values -> 512 bf16 bytes; absmax/quant_state are skipped.
    assert _bf16_model_bytes(write_fake_checkpoint(tmp_path)) == 512 + 400 + 100


def test_merge_device_follows_the_measured_facts():
    console = Console(file=io.StringIO())
    big = {"total_gib": 128.0, "available_gib": 100.0}
    small = {"total_gib": 16.0, "available_gib": 8.0}
    model = 10 * 2**30  # 10 GiB bf16 + ~4 GiB measured runtime overhead = 14 GiB needed
    assert _choose_merge_device(model, big, console) == {"": "cpu"}
    assert _choose_merge_device(model, small, console) == "balanced"


def test_merge_device_override():
    console = Console(file=io.StringIO())
    big = {"total_gib": 128.0, "available_gib": 100.0}
    small = {"total_gib": 16.0, "available_gib": 8.0}
    model = 10 * 2**30
    assert _choose_merge_device(model, big, console, device="gpu") == "balanced"
    assert _choose_merge_device(model, small, console, device="cpu") == {"": "cpu"}
