# rocm-qlora — QLoRA fine-tuning on AMD ROCm, built for learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm 7.1](https://img.shields.io/badge/ROCm-7.1-ED1C24)](https://www.amd.com/en/products/software/rocm.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

A **thin, readable wrapper** around the standard 2026 stack
(`transformers` + `TRL` + `PEFT` + `bitsandbytes`) for fine-tuning LLMs on
**AMD ROCm (gfx1100 / RDNA3, ROCm 7.1)**.

The training loop, quantization, chat templates and loss masking are all
handled by the libraries. This project adds exactly three things:

1. **Live observability** — one progress line per step (loss/EMA/trend,
   LR, grad norm, tokens/s, VRAM, ETA), `metrics.jsonl`, and matplotlib
   charts regenerated *while training runs*.
2. **AMD-specific correctness** — doctor-verified SDPA backends, RCCL
   transport configuration, allocator settings, a pre-quantized-checkpoint
   `prepare` step that makes 27B QLoRA fit on 24 GB cards.
3. **A clean output trail** — every run leaves a self-contained folder:
   config snapshot, environment fingerprint, per-step metrics, charts,
   per-epoch adapters, a Markdown report.

The *teaching* is the `docs/` set below — static, read deliberately, nothing
rendered into your terminal at runtime.

## House rules (the decisions behind the code)

* **Facts on the terminal, knowledge in `docs/`.** Commands print what
  happened and what to do next — plain status/progress lines, warnings,
  tables, artifact paths. Concepts, rationale and exact commands live in
  this README and `docs/`; no boxes, no teaching text at runtime.
* **One artifact per fact.** `metrics.jsonl` is the single source of
  truth: charts, `report.md` and the end-of-run numbers all derive from
  it, so nothing needs keeping in sync. The other files in a run dir are
  records of one thing each (config snapshot, environment fingerprint).
* **No fit prediction.** The launcher reports visible GPUs, free VRAM and
  the processes holding them, then launches. A run either fits or OOMs —
  the measured 24 GB budget is in `docs/02`, the knobs to react with in
  `docs/05`.
* **Configs explain themselves.** Every knob is commented where you set
  it (`configs/*.toml`); the reasoning behind the values is `docs/03`.
* **GPU-only, no CPU paths.** Model work (quantize, train, merge) runs on the
  GPUs — host RAM is staging for checkpoints too large for one card, nothing
  more. If a device set is too small, the run fails loudly; the numbers to
  react with are in `docs/02`/`docs/06`.
* **Loud guards, no silent rot.** Numerically broken training (NaN or
  exactly-zero loss, NaN `grad_norm`) aborts with a named `error` event
  and a doc pointer instead of continuing (`docs/05`).

## Requirements

* Linux with **AMD ROCm 7.1** user-space (the pinned torch wheel is built for it)
* One or more **gfx1100 / RDNA3** GPUs — multi-GPU runs use DDP automatically
* Python 3.12 via [uv](https://docs.astral.sh/uv/) — the interpreter and the
  whole stack (torch, TRL, PEFT, bitsandbytes, Triton) are pinned in
  `pyproject.toml` / `uv.lock`
* Disk: ~20 GB for the 27B checkpoint; host RAM matters only for `prepare`
  (staging a checkpoint that is far larger than one card)

## Quickstart

```bash
uv sync                                                     # .venv in-repo, torch from the ROCm 7.1 index
uv run rocm-qlora doctor                                    # probes GPUs, SDPA backends, bnb 4-bit
# dataset expected at .data/raw/dataset.json (configurable via data.path)
uv run rocm-qlora data --config configs/qwen3.8-27b.toml    # token-length profile
uv run rocm-qlora prepare --config configs/qwen3.8-27b.toml # NF4 checkpoint (once)
uv run rocm-qlora train --config configs/qwen3.8-27b.toml
```

Any knob can be overridden per run — `--set optim.lr=5e-5`, `--set
data.max_length=1280`, `--set model.attn=eager` — without editing the config
(see `--help`; docs/03 has the reasoning for each value).

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
| `configs/qwen3-8b.toml` | `Qwen/Qwen3-8B` | comfortable fallback, ~8 GB/GPU (derived from the parameter count — this config has not been run on this box), downloads on first use |
| `configs/toy.toml` | `Qwen/Qwen3.5-0.8B` | same architecture family as 3.8; pipeline rehearsal |

### Why `prepare` exists (the 27B memory story)

`load_in_4bit=True` on a **bf16 checkpoint** stages the entire bf16 model in
host RAM before quantizing (~55 GB for 27B — times the number of DDP ranks).
`prepare` quantizes **once** instead: it re-quantizes the modules upstream
left in 16-bit (`in_proj_qkv`, the vision tower — freeing ~3.7 GB/GPU: a
5 GB 16-bit projection becomes ~1.3 GB at NF4, the difference between
fitting and OOM on 24 GB cards), keeps `lm_head` 16-bit
(TRL's fused loss slices the raw weight tensor), and records **exact**
parameter accounting + provenance in `finetune_meta.json`. The
per-item 24 GB budget is in `docs/02-amd-rocm-notes.md`; in short: ~21 GB
measured at step 1, sawtoothing toward ~24 (docs/06).

### Speed: what's applied, what was rejected — and why

Applied (all measured): TRL's fused chunked head (full logits are never
materialized), AOTriton flash SDPA for the 16 full-attention layers, the
pre-quantized checkpoint, micro-batch 1 × accum 4 (same effective batch,
~half the activation memory of 2 × 2 — batch 2 × 1536 peaked at 25.5/25.75 GB
and OOMed, docs/05), `embed_tokens`/`lm_head` kept bf16, the DDP
init-broadcast skip (removes a redundant 4.9 GB transient), `adamw_8bit`,
non-reentrant gradient checkpointing.

Rejected, with evidence:

| option | why not |
|---|---|
| flash-linear-attention Triton kernels (the big lever for the 48 DeltaNet layers) | upstream issue #790: autotune configs of the exact kernel this model needs produce **wrong outputs**; #804: cold-start autotune can stall for an hour on gfx1100 (details in docs/02) |
| FP8 training (torchao/torchtitan) | CDNA3+ only; gfx1100 has no FP8 matrix cores |
| torchao NF4Tensor + torch.compile/FSDP2 | different ecosystem, ROCm maturity unclear; bnb NF4 is the supported path here |
| sequence packing | the DeltaNet layers have no document-aware masking → documents would attend to each other (silent data corruption) |
| `torch.compile` | bnb + hybrid attention + gradient checkpointing is not a supported combination here |
| Unsloth | single-GPU-centric, fights uv containment, caps torch below the ROCm 7.1 stack (docs/02) |

## What a run produces

```
.outputs/runs/<timestamp>-<name>/
├── charts/            loss, lr, grad_norm, throughput, vram, gpu_health, epochs, overview (.png, live-updated)
├── logs/rank-*.log
├── adapter/           final LoRA adapter + checkpoints (27B: every 42 steps)
├── metrics.jsonl      per-step records + events (the single source of truth)
├── report.md          auto-generated (numbers, per-epoch table, charts, events)
├── config.resolved.toml
└── env_fingerprint.json
```

## AMD/ROCm notes

Short version, all verified on RDNA3 hardware — full evidence in
`docs/02-amd-rocm-notes.md`:

* **Attention**: SDPA → AOTriton flash kernels work on gfx1100; `doctor`
  probes each backend in a fresh process and the launcher applies the
  verdict, disabling any backend the probe flagged (the mem-efficient
  *backward* is the observed intermittent-NaN source; padded batches then
  run math attention).
* **RCCL**: P2P works via dma-buf IPC (`HSA_ENABLE_IPC_MODE_LEGACY=0`,
  ~26 GB/s vs ~5.2 GB/s SHM); the default is still SHM — a simplicity choice,
  and the NaN events are model-side, not transport-side. Opt in with
  `[env.extra] NCCL_P2P_DISABLE = "0"`.
* **Hybrid attention**: the 48 DeltaNet layers run on transformers'
  pure-PyTorch fallbacks (the RDNA3 Triton path is the unstable part of the
  ecosystem).
* **Allocator**: `garbage_collection_threshold:0.8`; never enable
  `expandable_segments` (corrupted numerics on this stack, docs/05).
* **bitsandbytes**: 0.50.2 passes the training- and decode-shape 4-bit
  smoke tests (the ≤0.49.2 ROCm GEMV NaN is fixed).

## Repository layout

```
configs/                qwen3.8-27b.toml · qwen3-8b.toml · toy.toml
src/rocm_qlora/
  cli.py                all commands (doctor, data, prepare, train, report, chat, merge)
  config.py             TOML → pydantic models (strict: typos and bad types fail loudly)
  data.py               ShareGPT loading, validation, token profiling, previews
  model.py              prepare (NF4 + surgery) · 4-bit load · LoRA
  merge.py              adapter → bf16 merge (ready for GGUF export)
  metrics.py            metrics.jsonl: records, writer/reader, summaries, health guard
  callback.py           the training-time callback: progress lines, charts, GPU sampling
  plots.py              all matplotlib charts + Markdown report
  hardware.py           rocm-smi parsing · GPU selection
  probes.py             SDPA / RCCL-IPC / matmul probes
  launcher.py           GPU choice, worker env, torchrun orchestration
  sft.py                config → SFTConfig mapping (checkpoint cadence, warmup)
  train.py              the training worker (one process per GPU)
  doctor.py             environment + hardware report (.context artifacts)
  chat.py               interactive sanity check
  utils.py              formatting / timing / JSON helpers
docs/                   01-qlora · 02-amd-rocm-notes · 03-hyperparameters · 04-reading-the-plots · 05-troubleshooting · 06-27b-runbook
scripts/                 poc_p2p_allreduce.py (RCCL transport POC) · poc_deltanet_backward.py (27B NaN hunt)
tests/                   unit + smoke tests (stub tokenizers, synthetic metrics — no GPU)
```

**Before the 27B run**: `docs/06-27b-runbook.md` — exact commands,
expected timings/VRAM, what to watch, the open intermittent-NaN issue, the
GPU-reset/MES failure class, the mid-epoch checkpoint cadence (42 steps),
and the guard/resume behaviour.

## Code quality

One linter + formatter (ruff), one typo checker (codespell) and one type
checker (basedpyright), configured in `pyproject.toml` as dev-only
dependencies:

```bash
uv run ruff check .        # lint (E,W,F,I,UP,B,SIM,C4,PIE,RUF,PERF,RET,ISC)
uv run ruff format .       # format (120 cols, double quotes)
uv run codespell           # typos (skips .venv/.data/.outputs/.context)
uv run basedpyright        # type gate: 0 errors
uv run pytest              # unit + smoke tests, no GPU required
```

All five pass clean on this tree — treat a red run as a bug, not noise.
