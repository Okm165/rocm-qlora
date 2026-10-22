# 27B runbook — Qwen3.8-27B on a ShareGPT dataset

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
```

## What to expect

| | value |
|---|---|
| steps | `examples ÷ effective batch` per epoch × 3 epochs (effective batch = 2 × 2 accum × GPUs — it scales with the GPU count) |
| wall time | ~30–60 min total (compute-bound; the dashboard ETA row is ground truth) |
| VRAM | ~21–22 GB of 24 per GPU (weights 17.3 GB; rest is LoRA + activations + context) |
| eval | the 5% holdout at each epoch end (~2–3 min each) |
| artifacts | `charts/` refresh every 10 steps; `metrics.jsonl` has one record per step |

## What to watch (dashboard rows)

* **loss / EMA ↘** — expect ~1.3 → ~0.7–0.9 over 3 epochs on small Q&A data.
  The EMA is the honest signal; single-step wiggles are noise.
* **grad norm** — typically 0.3–1.5. Transient spikes are clipped and logged
  as `warn` events; a *sustained* rise is worth a look.
* **eval loss** — falling or flat = healthy; a clear rise = overfitting
  (fewer epochs, more dropout — `configs/qwen3.8-27b.toml` comments say where).
* **VRAM per GPU** — should sit flat around 21–22 GB. A slow climb means
  allocator fragmentation; resuming from the last checkpoint resets it.
* **health line** — temps < 80 °C, no warning events, loss trend not "rising".

## The guards (why the run aborts instead of rotting)

The monitor hard-stops on three numeric failure modes — learned from real
failures, see `docs/05-troubleshooting.md`:

1. non-finite loss (classic NaN/Inf spike),
2. **finite loss with NaN grad_norm** — the forward was fine, the backward
   poisoned the gradients; the next optimizer step would write NaN into
   every weight,
3. **loss exactly 0.0** — TRL's fused loss is `sum(-log_probs)/n` where
   ignored positions contribute zeros, so a poisoned model reports exactly
   zero while the loss EMA keeps "improving".

When one fires, the run exits with a loud error, an `error` event in `metrics.jsonl`, and
the run exits — nothing silent, nothing saved that looks healthy.

**Resume after an abort** (checkpoints are written every epoch, including
optimizer/scheduler state):

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

P2P/IPC **works** on RDNA3 workstations and is verified bit-exact at the 27B's
gradient sizes (`scripts/poc_p2p_allreduce.py`: ~25 GB/s vs ~5 GB/s for SHM, zero
mismatches over 100 iterations × 4 sizes on both transports). The *default* is
shared memory — a simplicity choice (keeps the IPC/IOMMU path out of the
picture) at ~45 ms/step, noise next to a ~10–25 s step. The NaN events below
are **not** transport-related.

To experiment with P2P anyway, add to the config:

```toml
[env.extra]
NCCL_P2P_DISABLE = "0"
```

## Known issue: intermittent NaN gradients on the 27B (open)

The 27B path intermittently produces a step with a *finite loss but
`grad_norm = NaN`* (~2 events in ~15 observed steps; the toy shows it rarely —
see docs/05). After such a step the optimizer poisons the weights and the
fused loss reports exactly 0.0 — which is why the guard exists and why it
will likely fire early in a real 27B run. It is **model-side, not
communication**: plain allreduce is bit-exact on both transports at the 27B's
gradient sizes.

**Root-cause lead (toy-scale)**: `ROCM_QLORA_DETECT_ANOMALY=1` pinned the
first non-finite backward op to `ScaledDotProductEfficientAttentionBackward0`
— the fused (mem-efficient) attention backward. A toy run forced to math
attention (`--set model.attn=eager`) completed 92/92 steps clean. On hardware
where the guard fires repeatedly, forcing math attention is the experiment to
try (slower — no fused kernel — but no fused backward either).

**What this means for a run**: the guard aborts cleanly (validated — it fired
at step 3 of a dry run), and `--resume` continues from the last epoch
checkpoint (validated on the toy). But with this rate, a full 3-epoch run
will likely need several resumes — before committing to that, spend one
small job on the root cause:

```bash
# 3 steps, ~4 min — names the exact op whose backward goes NaN
ROCM_QLORA_DETECT_ANOMALY=1 ROCM_QLORA_GRAD_CHECK=1 \
  uv run rocm-qlora train --config configs/qwen3.8-27b.toml --dry-run
```

`scripts/poc_deltanet_backward.py` isolates one Gated DeltaNet layer at the
27B's exact dimensions (needs a free GPU) for a targeted reproduction.

## If something goes wrong

* `docs/05-troubleshooting.md` — every failure seen so far, with its fix.
* The run dir is self-contained: `metrics.jsonl` (source of truth),
  `logs/rank-*.log`, `charts/`, `report.md`, `env_fingerprint.json`,
  `config.resolved.toml`.
* An aborted run leaves its last good checkpoint — resume from it rather than
  starting over.
