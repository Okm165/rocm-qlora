#!/usr/bin/env python
"""POC: one Qwen3.8 Gated DeltaNet layer at 27B dimensions — backward NaN hunt.

Context: the 27B training path intermittently produces NaN gradients with a
finite loss (observed at step 6 of a full run and step 3/3 of a dry run, over
both P2P and shared-memory transports — plain allreduce is bit-exact at the 27B
gradient sizes, see poc_p2p_allreduce.py). The toy model (same architecture,
smaller dims) shows the same failure ~20x more rarely. This isolates ONE
DeltaNet layer at the exact 27B dimensions with the same NF4 linears and the
same fp32 norms as training, and runs forward+backward in bf16 until it finds
a non-finite gradient (with autograd anomaly detection to name the op).

Usage:
    .venv/bin/python scripts/poc_deltanet_backward.py --iters 20 --seq 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_layer(prepared_dir: str, device: torch.device):
    """One Gated DeltaNet layer at the 27B's exact text-config dimensions."""
    import bitsandbytes as bnb
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = AutoConfig.from_pretrained(prepared_dir)
    text = getattr(cfg, "text_config", cfg)
    idx = next(i for i, t in enumerate(text.layer_types) if t == "linear_attention")
    layer = Qwen3_5GatedDeltaNet(text, layer_idx=idx).to(device, dtype=torch.bfloat16)

    # Match the prepared checkpoint: every Linear except lm_head is NF4;
    # norms/conv/A_log/dt_bias are fp32.
    n_quantized = 0
    for name, module in list(layer.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent = layer
            *path, attr = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            q = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            q.weight = bnb.nn.Params4bit(module.weight.data, requires_grad=False, quant_type="nf4")
            if module.bias is not None:
                q.bias = torch.nn.Parameter(module.bias.data)
            setattr(parent, attr, q.to(device))  # device move triggers NF4 quantization
            n_quantized += 1
    for module in layer.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            continue
        for param in module.parameters(recurse=False):
            if param.dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.float()
    print(f"layer {idx} ({text.layer_types[idx]}): {n_quantized} NF4 linears, rest fp32")
    return layer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=".data/models/qwen3.8-27b-nf4")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--scales", default="0.5,1.0,2.0,4.0")
    args = ap.parse_args()

    if not Path(args.prepared, "config.json").exists():
        print(f"prepared checkpoint not found at {args.prepared}")
        return 2

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    layer = build_layer(args.prepared, device)
    hidden = layer.hidden_size
    layer.train()

    scales = [float(s) for s in args.scales.split(",")]
    failures = 0
    for it in range(args.iters):
        scale = scales[it % len(scales)]
        torch.manual_seed(it)
        x = (torch.randn(1, args.seq, hidden, device=device, dtype=torch.bfloat16) * scale).contiguous()
        layer.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(x)
                loss = out.float().square().mean()
            loss.backward()
        except RuntimeError as exc:  # anomaly detection raises here
            failures += 1
            print(f"iter {it} (scale {scale}): FAIL — {str(exc).splitlines()[0][:160]}")
            continue
        bad = [name for name, p in layer.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            failures += 1
            print(f"iter {it} (scale {scale}): non-finite grads in {bad[:4]}")
        elif it < 3 or it % 5 == 0:
            print(f"iter {it} (scale {scale}): ok  loss={loss.item():.4f}")

    print(json.dumps({"iters": args.iters, "seq": args.seq, "failures": failures}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
