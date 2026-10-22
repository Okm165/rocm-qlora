# 03 · Hyperparameters — every knob, with reasoning

The config files carry inline comments; this doc explains *why* the values
are what they are. Ranges are the 2026 practitioner consensus (QLoRA paper,
Unsloth guides, TRL docs, field reports), not laws of nature.

## LoRA

| knob | value here | range / guidance |
|---|---|---|
| `r` (rank) | 16 | 8 for format-only, 16–32 general SFT, 64+ for heavy rewriting. On **small data prefer lower capacity** — the bottleneck is a regularizer against memorization. |
| `alpha` | 32 | convention `α = 2r` (scaling `α/r = 2`). Doubling `r` while holding `α` halves the effective step size — hold `α/r` constant when sweeping rank. |
| `dropout` | 0.05 | 0–0.1. Small datasets benefit from mild dropout; above ~50k examples it stops mattering. |
| `target_modules` | explicit list (all text-side linears) | **All linear layers** — QLoRA ablations show this matters more than rank. The explicit list (instead of `all-linear`) excludes the vision tower and the output head. |
| `bias` | none | keep frozen; biases are tiny and noisy on small data. |

## Optimization

| knob | value | reasoning |
|---|---|---|
| `lr` | 1e-4 (27B), 2e-4 (toy) | QLoRA range 1e-4–2e-4 for 7–13B; conservative end for a 27B that is already close to the target behavior. LoRA LRs run ~10× higher than full-fine-tune LRs. |
| `scheduler` | cosine | explore early, refine late. `constant` (the QLoRA paper) also works; `linear` is a fine third option. |
| `warmup_ratio` | 0.03 | Adam's second-moment estimate is garbage for the first steps; 3–10% warmup is standard. (Computed into `warmup_steps` internally — transformers v5 removed the ratio.) |
| `max_grad_norm` | 0.3 | the QLoRA paper value; aggressively clips outlier batches on small data. |
| `weight_decay` | 0.01 | mild; the base is frozen, this only shapes the adapter. |
| `optimizer` | `adamw_8bit` (27B) / `adamw_torch` (8B, toy) | 8-bit states save ~1 GB at 27B scale; plain AdamW is simplest when VRAM allows. **Paged optimizers are CUDA-only** — the config validator rejects them with a hint. |
| `epochs` | 3 | 1–3 for instruction data; up to 5 only if eval loss keeps improving. Watch the train/eval gap. |
| `per_device_batch_size × grad_accum × GPUs` | e.g. 2 × 2 × N GPUs | a few dozen steps/epoch on a few hundred examples. For small data, effective batch 8–32 keeps enough optimizer steps for the schedule to mean anything. |
| `gradient_checkpointing` | on | recompute activations instead of storing them: ~30% slower steps, several GB saved. Uses the non-reentrant path (`use_reentrant=False`). |
| `loss_type` | `nll` | TRL 1.16 computes every loss through a **fused chunked head** that never materializes full logits (this is the standard path — `chunked_nll` is just a deprecated alias). `dft` switches the weighting math (up-weights uncertain tokens). |

## Data

| knob | value | reasoning |
|---|---|---|
| `max_length` | 1536 | p99 of the measured token distribution (see `rocm-qlora data`). Setting it from the model's 262k context window would waste memory on every batch; setting it from the observed max buys nothing. |
| `assistant_only_loss` | true | gradient flows only from assistant tokens (reasoning + answer). User tokens are masked with -100. TRL auto-patches Qwen3.8's template with `{% generation %}` markers. |
| `packing` | **false** | packing is a throughput win on short data, but a **correctness hazard**: without document-aware attention masks, tokens can attend across example boundaries. With a few hundred examples, padding waste costs seconds. Enable it only with a clear understanding of the attention backend's boundary handling. |
| `eval_fraction` | 0.05 | a small holdout (a few dozen examples). Enough to see the *shape* of generalization; not enough to read decimals. |

## Quantization (prepare step)

| knob | value | reasoning |
|---|---|---|
| `quant_type` | nf4 | normal-distribution-matched 4-bit; the QLoRA default. |
| `double_quant` | true | quantize the quantization scales; ~0.4 bits/param cheaper. |
| `quant_storage` | uint8 | required for FSDP/sharding compatibility and what upstream checkpoints use. |
| `requantize_modules` | `["in_proj_qkv"]` | the upstream checkpoint left this 2.5B-param projection in 16-bit (~5 GB/GPU). Re-quantizing it is the single biggest memory win. `lm_head` is deliberately **not** here (chunked-CE constraint). |
| `requantize_vision` | true | the vision tower is unused for text training; quantizing it saves ~1 GB. |

## What to change when something is off

| symptom | first knob to try |
|---|---|
| loss flat for many steps | ↑ `lr` (×2), ↓ `dropout` |
| loss spiky, grad norm exploding | ↓ `lr` (÷2), ↑ `grad_accum` |
| eval loss rising while train falls | ↓ `epochs`, ↑ `dropout`, ↓ `r` |
| OOM on the 27B | ↓ `max_length` (1280/1024), or drop `per_device_batch_size` to 1 |
| too few steps/epoch (loss unreadable) | ↓ `grad_accum` |
| steps too slow | ↑ `per_device_batch_size` if VRAM allows (1→2) |
