# rocm-qlora

QLoRA fine-tuning for **AMD ROCm** GPUs (RDNA3 / gfx1100 class): a thin layer
around `transformers`, `TRL`, `PEFT` and `bitsandbytes`. Text-only supervised
fine-tuning, GPU-only (there is no CPU path).

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ROCm 7.1](https://img.shields.io/badge/ROCm-7.1-ED1C24)](https://www.amd.com/en/products/software/rocm.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

The training loop, quantization, chat templates and loss masking come from the
libraries. This tool adds:

* **Live observability** — one compact progress line per step, `metrics.jsonl`
  as the single source of truth, charts regenerated while training runs.
* **Working AMD defaults** — the SDPA/AOTriton, RCCL and allocator settings
  measured on gfx1100, exposed as commented config knobs instead of hidden
  logic (`docs/02`).
* **A `prepare` step** — re-quantizes a checkpoint once so 27B QLoRA fits on
  24 GB cards.
* **Loud failure modes** — NaN/zero-loss guards, run-directory collision
  guards, unknown numbers rendered as `n/a` — never as zero.

## Requirements

* Linux with **AMD ROCm 7.x** user-space
* One or more **gfx1100 / RDNA3** GPUs — multi-GPU runs use DDP automatically
* Python 3.12 with [uv](https://docs.astral.sh/uv/); the whole stack (torch,
  TRL, PEFT, bitsandbytes, Triton) is pinned in `pyproject.toml` / `uv.lock`
* Disk: ~20 GB for a 27B checkpoint; host RAM matters only for `prepare`
  (staging a checkpoint far larger than one card)

## Quickstart

```bash
uv sync
uv run rocm-qlora doctor                                  # hardware + stack report
# dataset: ShareGPT-style JSON; `data.path` in the config (default .data/raw/dataset.json)
uv run rocm-qlora data    --config configs/qwen3.8-27b.toml   # token-length profile
uv run rocm-qlora prepare --config configs/qwen3.8-27b.toml   # NF4 checkpoint (once per base model)
uv run rocm-qlora train   --config configs/qwen3.8-27b.toml
```

Rehearse the whole pipeline in about two minutes with the toy model (same
architecture family, tiny weights):

```bash
uv run rocm-qlora prepare --config configs/toy.toml
uv run rocm-qlora train   --config configs/toy.toml --dry-run
```

After a run:

```bash
uv run rocm-qlora chat   --run .outputs/runs/<id>   # talk to the adapter
uv run rocm-qlora merge  --run .outputs/runs/<id>   # bf16 merge, ready for GGUF export
uv run rocm-qlora report --run .outputs/runs/<id>   # regenerate charts + report
```

Any knob can be overridden per run without editing the config — `--set
optim.lr=5e-5`, `--set data.max_length=1280`, `--set model.attn=eager`.

## Configs

| config | base model | notes |
|---|---|---|
| `configs/qwen3.8-27b.toml` | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | 27B dense hybrid (Gated DeltaNet + full attention); ~21 GB/GPU at step 1, peaks ~25.6 of a 24 GiB card |
| `configs/qwen3-8b.toml` | `Qwen/Qwen3-8B` | comfortable fallback, ~8 GB/GPU (estimated); downloads on first use |
| `configs/toy.toml` | `Qwen/Qwen3.5-0.8B` | same architecture family, tiny weights — pipeline rehearsal |

Every knob is commented where you set it; the reasoning behind the values is
in `docs/03`.

## Why `prepare` exists

`load_in_4bit` on a bf16 checkpoint stages the whole bf16 model in host RAM
before quantizing (~55 GB for 27B, times the number of DDP ranks). `prepare`
quantizes **once** instead: it re-quantizes the modules upstream left in
16-bit (for the 27B config this frees ~3.7 GB/GPU), keeps `lm_head` 16-bit
(TRL's fused loss slices the raw weight tensor) and records exact parameter
accounting + provenance in `finetune_meta.json`.

## What a run produces

```
.outputs/runs/<timestamp>-<name>/
├── charts/            loss, lr, grad_norm, throughput, vram, gpu_health, epochs, overview (.png, live-updated)
├── logs/rank-*.log
├── adapter/           final LoRA adapter + checkpoints
├── metrics.jsonl      per-step records + events (the single source of truth)
├── report.md          auto-generated (numbers, per-epoch table, charts, events)
├── config.resolved.toml
└── env_fingerprint.json
```

## Documentation

| doc | contents |
|---|---|
| `docs/00-design.md` | what to expect from this tool (design notes) |
| `docs/01-qlora.md` | the method: NF4, double quantization, LoRA, loss masking |
| `docs/02-amd-rocm-notes.md` | measured gfx1100 facts: attention, RCCL, allocator, bitsandbytes, rejected options |
| `docs/03-hyperparameters.md` | every knob, and why the shipped defaults are what they are |
| `docs/04-reading-the-plots.md` | the chart decoder |
| `docs/05-troubleshooting.md` | symptoms → causes → knobs |
| `docs/06-27b-runbook.md` | the end-to-end 27B recipe: commands, timings, guards |

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run codespell && uv run basedpyright && uv run pytest
```

All five pass on a clean checkout; a red run is a bug, not noise.
