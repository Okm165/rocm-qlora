# rocm-qlora — QLoRA fine-tuning on AMD ROCm, built for learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm 7.1](https://img.shields.io/badge/ROCm-7.1-ED1C24)](https://www.amd.com/en/products/software/rocm.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

A **thin, teaching-first wrapper** around the standard 2026 stack
(`transformers` + `TRL` + `PEFT` + `bitsandbytes`) for fine-tuning LLMs on
**AMD ROCm (gfx1100 / RDNA3, ROCm 7.1)**.

The training loop, quantization, chat templates and loss masking are all
handled by the libraries. This project adds exactly four things:

1. **A short explainer set** — the handful of concepts that actually matter
   here (QLoRA, the prepared checkpoint, ROCm attention, loss reading) shown
   in the terminal *when they first matter*, with links into `docs/`.
2. **A live instrument panel** — rich dashboard (loss/EMA/trend, LR, grad
   norm, tokens/s, ETA, per-GPU VRAM/temp/power) + `metrics.jsonl` +
   matplotlib charts regenerated *while training runs*.
3. **AMD-specific correctness** — doctor-verified SDPA backends, RCCL
   transport configuration, allocator settings, a pre-quantized-checkpoint
   `prepare` step that makes 27B QLoRA fit on 24 GB cards.
4. **A clean output trail** — every run leaves a self-contained folder:
   config snapshot, environment fingerprint, per-step metrics, charts,
   per-epoch adapters, a Markdown report with interpretation.

## Requirements

* Linux with **AMD ROCm 7.1** user-space (the pinned torch wheel is built for it)
* One or more **gfx1100 / RDNA3** GPUs — multi-GPU runs use DDP automatically
* Python 3.12 via [uv](https://docs.astral.sh/uv/) — the interpreter and the
  whole stack (torch, TRL, PEFT, bitsandbytes, Triton) are pinned in
  `pyproject.toml` / `uv.lock`
* Disk: ~20 GB for the 27B checkpoint; host RAM only matters for `prepare`

## Quickstart

```bash
uv sync                                                     # .venv in-repo, torch from the ROCm 7.1 index
uv run rocm-qlora doctor                                    # probes GPUs, SDPA backends, bnb 4-bit
# dataset expected at .data/raw/dataset.json (configurable via data.path)
uv run rocm-qlora data --config configs/qwen3.8-27b.toml    # token-length profile
uv run rocm-qlora prepare --config configs/qwen3.8-27b.toml # NF4 checkpoint (once)
uv run rocm-qlora train --config configs/qwen3.8-27b.toml
```

Then:

```bash
uv run rocm-qlora chat --run .outputs/runs/<id>   # talk to the adapter
uv run rocm-qlora merge --run .outputs/runs/<id>  # bf16 merge → GGUF for llama.cpp
uv run rocm-qlora report --run .outputs/runs/<id> # regenerate charts + report
```

Validate the whole pipeline in ~4 minutes with the toy model (same hybrid
architecture, tiny weights):

```bash
uv run rocm-qlora prepare --config configs/toy.toml
uv run rocm-qlora train  --config configs/toy.toml --dry-run
```

## The models

| config | model | notes |
|---|---|---|
| `configs/qwen3.8-27b.toml` | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | the heavy config: 27B dense hybrid (Gated DeltaNet + full attention), ~21–24 GB/GPU measured (batch 1; peaks near the 24 GB ceiling) |
| `configs/qwen3-8b.toml` | `Qwen/Qwen3-8B` | comfortable fallback, ~8 GB/GPU, downloads on first use |
| `configs/toy.toml` | `Qwen/Qwen3.5-0.8B` | same architecture family as 3.8; pipeline rehearsal |

### Why `prepare` exists (the 27B memory story)

`load_in_4bit=True` on a **bf16 checkpoint** stages the entire bf16 model in
host RAM before quantizing (~55 GB for 27B — times the number of DDP ranks).
`prepare` instead:

1. loads the **pre-quantized** upstream checkpoint (~22 GB, mostly NF4),
2. **re-quantizes the modules upstream left in 16-bit** (`in_proj_qkv`, the
   vision tower) — worth ~5 GB per GPU, the difference between fitting and
   OOM on 24 GB cards,
3. keeps `lm_head` in 16-bit (TRL 1.16 computes every SFT loss through a
   fused chunked head that slices the raw weight tensor — a quantized head
   breaks it),
4. writes a `finetune_meta.json` with **exact** parameter accounting, which
   the VRAM estimator and the run preflight use.

### Per-GPU VRAM budget (27B, DDP, measured estimates)

| component | GB |
|---|---|
| NF4 weights + 16-bit embeddings/lm_head | ~17.3 (exact accounting in finetune_meta.json) |
| LoRA (r=16, all-linear) + 8-bit Adam | ~0.7 |
| activations @ 1536 tokens, batch 1, gradient checkpointing | ~1.3 |
| HIP context + fragmentation | ~1 |
| **total** | **~21 of 24** (measured 21.1 at step 1; sawtooths toward ~24 — docs/06) |

### Speed: what's applied, what was rejected — and why

Applied (all measured):

| win | effect |
|---|---|
| TRL 1.16 fused chunked head | full logits are never materialized (the "Cut Cross-Entropy" idea, built in) |
| AOTriton flash SDPA | 16 full-attention layers run flash kernels |
| RCCL dma-buf IPC (`HSA_ENABLE_IPC_MODE_LEGACY=0`) | ~26 GB/s vs ~5.2 GB/s SHM collectives (bit-exact at 27B sizes; SHM kept as the default — see the notes below) |
| pre-quantized checkpoint (`prepare`) | no 55 GB on-the-fly quantization pass at load |
| micro-batch 1 × accum 4 (same effective batch) | ~half the activation memory of 2 × 2 — the 1536-token batch-2 runs peaked at 25.5/25.75 GB and OOMed (see docs/05); batch 2 is faster when it fits |
| `embed_tokens`/`lm_head` stay bf16 | step memory 25 → ~20 GB; the fit comes from batch 1 × accum 4 (batch 2 × 1536 peaked 25.5–25.75 GB and OOMed — docs/05) |
| DDP init-broadcast skip | removes a redundant 4.9 GB transient (identical checkpoints + shared seed ⇒ replicas identical by construction) |
| `adamw_8bit`, non-reentrant gradient checkpointing, 8 dataloader workers | standard hygiene |

Rejected, with evidence:

| option | why not |
|---|---|
| flash-linear-attention Triton kernels (the big remaining lever for the 48 DeltaNet layers) | FLA #790: certain autotune configs of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` — the exact kernel this model needs — produce **wrong outputs** (training diverged); #804: cold-start Triton stalls on ROCm (82 min autotune reported on gfx1100). Pure-PyTorch fallback stays. |
| FP8 training (torchao/torchtitan) | CDNA3+ only (MI300/MI325); gfx1100 has no FP8 matrix cores → emulation, slower |
| torchao NF4Tensor + torch.compile/FSDP2 | different ecosystem, ROCm maturity unclear; bnb NF4 is the supported path here |
| sequence packing | the DeltaNet layers have no document-aware masking → documents would attend to each other (silent data corruption). Off until that's solved. |
| `torch.compile` | bnb + hybrid attention + gradient checkpointing is not a supported combination here |
| Unsloth | fastest on paper, but single-GPU-centric, fights uv containment, caps torch below the ROCm 7.1 stack. Kept as a documented alternative (docs/02). |

## What a run produces

```
.outputs/runs/<timestamp>-<name>/
├── charts/            loss, eval, lr, grad_norm, throughput, vram, gpu_health, epochs, dashboard (.png, live-updated)
├── logs/rank-*.log
├── adapter/           final LoRA adapter + checkpoints (27B: every 42 steps)
├── metrics.jsonl      one JSON record per optimizer step (the single source of truth)
├── summary.json       headline numbers
├── report.md          auto-generated with interpretation + next steps
├── config.resolved.toml / .json
└── env_fingerprint.json
```

## AMD/ROCm notes (verified on RDNA3 hardware)

* **Attention**: PyTorch SDPA dispatches to AOTriton flash kernels on gfx1100
  when `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; `doctor` probes each
  backend in a fresh process (deferred launch errors otherwise poison the
  process) and the launcher applies the verdict automatically — including
  disabling any backend the probe flagged as broken (on gfx1100 the
  mem-efficient backward is the observed intermittent-NaN source; padded
  batches then run math attention).
* **RCCL**: consumer RDNA3 P2P works once the runtime uses **dma-buf IPC**
  (`HSA_ENABLE_IPC_MODE_LEGACY=0`; the default legacy mode needs a DKMS-only
  ioctl and fails on mainline kernels with `hipIpcGetMemHandle: invalid
  argument`). Measured ~26 GB/s vs ~5.2 GB/s for shared memory, and verified
  bit-exact at the 27B's gradient sizes on **both** transports
  (`scripts/poc_p2p_allreduce.py`). The default is SHM
  (`NCCL_P2P_DISABLE=1`) — a simplicity choice at ~45 ms/step for LoRA; the
  intermittent NaN-grad events seen on the 27B are model-side, not
  transport-related (the fused attention backward is the lead — see the
  runbook). Opt in with
  `[env.extra] NCCL_P2P_DISABLE = "0"`.
* **bitsandbytes**: 0.50.2 passes both training-shape and decode-shape
  4-bit smoke tests (the ≤0.49.2 ROCm GEMV NaN bug is fixed).
* **Hybrid attention**: Qwen3.8's Gated DeltaNet layers run on transformers'
  pure-PyTorch fallbacks (no flash-linear-attention/Triton kernels — the
  RDNA3 Triton path is the unstable part of the ecosystem).
* **Allocator**: `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`
  (no `expandable_segments` — tried 2026-10-23, corrupted numerics; see docs/05).

## Repository layout

```
configs/                qwen3.8-27b.toml · qwen3-8b.toml · toy.toml
src/rocm_qlora/
  cli.py                all commands (doctor, data, prepare, train, report, chat, merge)
  config.py             TOML → validated dataclasses (strict: typos fail loudly)
  data.py               ShareGPT loading, validation, token profiling, previews
  model.py              prepare (NF4 + surgery) · 4-bit load · LoRA · merge
  monitor.py            metrics.jsonl · live dashboard · health checks · summaries
  plots.py              all matplotlib charts + Markdown report
  hardware.py           rocm-smi parsing · VRAM estimator · GPU selection · SDPA probe
  launcher.py           GPU choice, worker env, torchrun orchestration
  train.py              the training worker (one process per GPU)
  explain.py            contextual explainer panels (deduplicated per run)
  chat.py               interactive sanity check
docs/                   01-qlora-explained · 02-amd-rocm-notes · 03-hyperparameters · 04-reading-the-plots · 05-troubleshooting · 06-27b-runbook
scripts/                 train_worker.py · poc_p2p_allreduce.py (RCCL transport POC) · poc_deltanet_backward.py (27B NaN hunt)
tests/                   62 tests, none needs a GPU
```

**Before the 27B run**: `docs/06-27b-runbook.md` — exact commands,
expected timings/VRAM, what to watch, the open intermittent-NaN issue, the
GPU-reset/MES failure class, the mid-epoch checkpoint cadence (42 steps),
and the guard/resume behaviour.

## Code quality

One linter + formatter (ruff) and one typo checker (codespell), configured in
`pyproject.toml`, all dev-only dependencies:

```bash
uv run ruff check .        # lint (E,W,F,I,UP,B,SIM,C4,PIE,RUF,PERF,RET,ISC)
uv run ruff format .       # format (120 cols, double quotes)
uv run codespell           # typos (skips .venv/.data/.outputs/.context)
uv run pytest              # 26 tests, no GPU required
```

All four pass clean on this tree — treat a red run as a bug, not noise.
