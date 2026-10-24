# 01 · QLoRA explained (for this exact project)

## The three ideas

**1. NF4 quantization.** The frozen base model's Linear weights are stored in
4 bits (~0.55 bytes/param including quantization scales) instead of 16. NF4
places its 16 code values at the quantiles of a standard normal distribution —
information-theoretically optimal for the roughly-Gaussian weights of trained
transformers. The matmul itself still runs in **bfloat16**: weights are
dequantized on the fly per tile. Memory drops ~4×; compute gets a small
kernel overhead.

**2. Double quantization.** The per-block quantization scales (`absmax`) are
themselves quantized (to 8-bit with their own scales). Saves ~0.4 bits/param
more — on a 27B model that's ~1.3 GB.

**3. LoRA adapters.** Instead of updating `W` (27B params), LoRA trains
`ΔW = (α/r)·B·A` where `B` (out×r) and `A` (r×in) are tiny. Only `A` and `B`
receive gradients. For this project: `r=16` on every text-side linear layer →
~0.5% of parameters trainable, optimizer state in the hundreds of MB instead
of tens of GB.

Why this preserves quality: the frozen base cannot be catastrophically
forgotten, and on small datasets the low-rank bottleneck acts as a
*regularizer* — a few-hundred-example dataset would happily overfit a full fine-tune.

## What the pipeline does, step by step

```
prepare (once)
  source checkpoint (pre-quantized NF4 from the community, or plain bf16)
    → re-quantize modules the upstream left in 16-bit        [model.py]
    → write a self-describing quantization_config            [model.py]
    → save + finetune_meta.json (exact parameter accounting) [model.py]

train (each run)
  prepared checkpoint → 4-bit load onto this rank's GPU      [transformers+bnb]
  selective fp32 cast (norms yes, embeddings no) + freeze    [model.py]
  LoRA attach (all text-side linears; vision dropped, lm_head excluded) [peft]
  TRL SFTTrainer:
    chat template renders user→assistant turns               [transformers]
    assistant-only loss mask ({% generation %} markers)      [trl]
    chunked cross-entropy (no fp32 logits materialization)   [trl]
    bf16 autocast + gradient checkpointing                   [transformers]
  DDP across the GPUs, SHM collectives                       [torchrun+rccl]
```

Everything else in this repo is *observation*: progress lines, JSONL, charts,
health checks, report.

## The dataset (what to point `data.path` at)

* Any ShareGPT-style JSON: `messages: [{role, content}, ...]` turns, single or
  multi-turn. `rocm-qlora data` validates it loudly before any GPU work.
* Assistant turns may contain ` thinking…</think>` reasoning traces — the loss
  is computed on **both** (reasoning + answer), so the model learns the
  *thinking style*, not just input→output mappings. Qwen3.8's chat template
  parses ` thinking` natively, and TRL's bundled Qwen3.8 training template adds
  the loss masks.
* Token lengths drive `max_length` — measure them with `rocm-qlora data`.
  Typical Q&A data lands around p50 ≈ 500 and p99 ≈ 1.5k tokens; pick the cap
  where the coverage curve flattens.
* For scale: a few hundred thousand tokens → a few dozen optimizer steps per
  epoch at effective batch 12.

## Expectations for a small-dataset run (r=16, 3 epochs)

| signal | healthy | worrying |
|---|---|---|
| loss at step 0 | ~0.8–2.0 (in-domain, strong base) | >3 (template/formatting bug) |
| loss after 1 epoch | −0.1 to −0.4 from start | flat (LR too low / no learning) |
| eval − train | ≤ 0.15 | rising eval gap (overfitting) |
| grad norm | 0.1–2 with occasional spikes | sustained >10 (LR too high) |
| tokens/s | stable in the progress lines | decaying (thermal throttle / leak) |

What not to expect: new knowledge injection. A few hundred examples
steer style, format, vocabulary and reasoning structure; they do not teach
new facts. Capability injection needs 100× the data or continued
pretraining.

## Read more

- QLoRA paper — https://arxiv.org/abs/2305.14314
- bitsandbytes 4-bit docs — https://huggingface.co/docs/bitsandbytes
- PEFT LoRA docs — https://huggingface.co/docs/peft
- TRL SFTTrainer — https://huggingface.co/docs/trl/sft_trainer
