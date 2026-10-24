# 06 · 27B runbook — Qwen3.8-27B on a ShareGPT dataset

Everything is already configured in `configs/qwen3.8-27b.toml`. This page is
the operator's view: what to run, what to expect, what to watch, and what to
do when something looks wrong. The 27B run is the heavy one; the same
commands rehearse the full code path on `configs/toy.toml` in ~4 minutes.

## The commands, in order

```bash
# 0. one-time environment sanity (SDPA verdict, bnb smoke, RCCL IPC probe)
uv run rocm-qlora doctor

# 1. build the slimmed NF4 checkpoint (~50 s; needed once, or after config changes)
uv run rocm-qlora prepare --config configs/qwen3.8-27b.toml

# 2. dry run — 3 steps on 8 examples (~3 min): proves load → step → save → charts
uv run rocm-qlora train --config configs/qwen3.8-27b.toml --dry-run

# 3. the real run (expect ~30–60 min wall time)
uv run rocm-qlora train --config configs/qwen3.8-27b.toml

# 4. regenerate charts + report from metrics.jsonl at any time
uv run rocm-qlora report --run .outputs/runs/<timestamp>-qwen3.8-27b

# 5. talk to the adapter (base + LoRA) as a sanity check
uv run rocm-qlora chat --run .outputs/runs/<timestamp>-qwen3.8-27b

# 6. fold the adapter into a bf16 model (~55 GB, sharded across the GPUs —
#    needs ~55 GB of aggregate free VRAM; fails loudly if the device set is
#    too small), then export GGUF
uv run rocm-qlora merge --run .outputs/runs/<timestamp>-qwen3.8-27b
python llama.cpp/convert_hf_to_gguf.py .outputs/runs/<run>/merged \
    --outfile model-f16.gguf --outtype f16
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

## What to expect

| | value |
|---|---|
| steps | `examples ÷ effective batch` per epoch × 3 epochs (effective batch = 1 × 4 accum × GPUs — it scales with the GPU count) |
| wall time | ~75–90 min total (measured ~18–21 s/step × 252 steps + 3 evals; the ETA on the progress line is ground truth) |
| VRAM | ~21–24 GB of 24 per GPU (step 1 ≈ 21.1 GB; peaks ~24 — weights 17.7 GB, rest is LoRA + activations + context) |
| eval | the 5% holdout at each epoch end (~2–3 min each) |
| artifacts | `charts/` refresh every 10 steps; `metrics.jsonl` has one record per step |

## What to watch (progress line + charts)

* **loss / EMA ↘** — expect ~1.3 → ~0.7–0.9 over 3 epochs on small Q&A data.
  The EMA is the honest signal; single-step wiggles are noise.
* **grad norm** — typically 0.3–1.5. A transient spike (> 10× the recent
  median) is clipped and logged as a `warn` event, at most once per 50 steps;
  a *sustained* rise is worth a look.
* **eval loss** — falling or flat = healthy; a clear rise = overfitting
  (fewer epochs, more dropout — `configs/qwen3.8-27b.toml` comments say where).
* **VRAM per GPU** — starts ~21 GB and sawtooths toward ~24 GB (2026-10-23
  batch-1 run); a sustained climb means fragmentation — resuming resets it.
* **warnings** — the console prints `!` lines when the loss EMA rises, a GPU
  hits 90 °C, or VRAM grows by > 1.5 GB since step 10; each family fires at
  most once per 100 steps, and none should appear on a healthy run.

## The guards (why the run aborts instead of rotting)

The monitor hard-stops on three numeric failure modes — non-finite loss,
finite loss with NaN `grad_norm`, and loss exactly 0.0 — writes an `error`
event, and exits loudly; nothing silent, nothing saved that looks healthy.
The full story of each mode is in `docs/05-troubleshooting.md`.

**Resume after an abort** (checkpoints carry optimizer/scheduler state; the
27B config writes one every 42 steps — half an epoch — so a crash costs at
most ~11 min):

```bash
uv run rocm-qlora train --config configs/qwen3.8-27b.toml \
    --resume .outputs/runs/<timestamp>-qwen3.8-27b
```

The latest `adapter/checkpoint-*` is picked automatically; the new run dir
records the resumed segment.

**Hunting a NaN further** — two env-gated diagnostics:

```bash
ROCM_QLORA_DETECT_ANOMALY=1 ROCM_QLORA_GRAD_CHECK=1 \
  uv run rocm-qlora train --config configs/qwen3.8-27b.toml
```

`DETECT_ANOMALY` makes autograd name the exact op whose backward produced the
NaN (with a traceback); `GRAD_CHECK` installs per-parameter hooks that name
the exact tensor, before the optimizer step. Both are diagnostic-only (slower).

## Transport note (P2P vs shared memory)

The default is shared memory — a simplicity choice; the NaN events are
**not** transport-related (both transports are bit-exact at the 27B's
gradient sizes). To experiment with P2P, add `[env.extra] NCCL_P2P_DISABLE =
"0"` to the config. Full evidence in `docs/02-amd-rocm-notes.md`.

## Known issue: intermittent NaN gradients on the 27B (open)

The 27B path intermittently produces a step with a *finite loss but
`grad_norm = NaN`*; after such a step the optimizer poisons the weights and
the fused loss reports exactly 0.0 — which is why the guard exists and why it
may fire early in a real 27B run. It is **model-side, not communication**.
The root-cause lead, the measured rates and the diagnostic run are in
`docs/05-troubleshooting.md` ("grad_norm is NaN"); the env-gated diagnostics
above name the exact op / parameter when it happens.

**What this means for a run**: the guard aborts cleanly (validated — it fired
at step 3 of a dry run) and `--resume` continues from the last checkpoint
(validated on the toy). With this rate, a full 3-epoch run may need several
resumes.

## If something goes wrong

* `docs/05-troubleshooting.md` — every failure seen so far, with its fix.
* A **GPU reset** (`MES failed to respond` in the kernel log; all ranks die
  with SIGABRT) is a firmware/hardware event, not a code bug — diagnosis,
  mitigations and the firmware check are in docs/05 ("GPU reset: MES failed
  to respond"). Mid-epoch checkpoints make it survivable.
* The run dir is self-contained: `metrics.jsonl` (source of truth),
  `logs/rank-*.log`, `charts/`, `report.md`, `env_fingerprint.json`,
  `config.resolved.toml`.
* An aborted run leaves its last good checkpoint — resume from it rather than
  starting over.
