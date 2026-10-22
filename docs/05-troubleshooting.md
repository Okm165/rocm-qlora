# 05 · Troubleshooting (real failures, real fixes)

Every entry below is a real failure or finding from building this project on
gfx1100 / ROCm 7.1 — not generic advice.

## Where to look first

```
<run>/logs/rank-0.log      the full story (tracebacks, warnings)
<run>/metrics.jsonl        the numeric story (events have type="event")
.context/doctor-latest.md  machine state at run start
```

---

## "ncclUnhandledCudaError: invalid argument" at the first barrier

**Cause (root-caused and fixed):** the ROCm 7.1 runtime defaults to *legacy*
IPC mode, whose handle-export ioctl exists only in the out-of-tree DKMS
driver — on mainline kernels `hipIpcGetMemHandle` fails and RCCL's P2P
transport dies.

**Fix (already default in the launcher):** `HSA_ENABLE_IPC_MODE_LEGACY=0`
selects the dma-buf IPC path; P2P then works (~26 GB/s measured vs ~5.2 GB/s
for shared memory). Diagnose with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P` — the log should show channels
`via P2P/IPC`.

If a driver/stack update reintroduces hangs or `illegal-state` errors, fall
back with `NCCL_P2P_DISABLE=1` in the config's `[env.extra]`. Do **not** add
`iommu=pt` (translated IOMMU works; one upstream report shows `iommu=pt`
corrupting peer IPC).

## "No available kernel" / attention crashes mid-run

RDNA3 SDPA backends can fail in confusing ways — the error may surface at an
*unrelated* op later (deferred launch failure). Run:

```bash
uv run rocm-qlora doctor          # probes FLASH / EFFICIENT / MATH in fresh processes
```

The launcher then picks `sdpa` or `eager` from the verdict automatically. If
fused attention dies after a driver/stack change, force math attention:

```bash
uv run rocm-qlora train --set model.attn=eager --config …
```

## NaN loss at step 0

1. `lm_head` quantized → shape errors, not NaN — see the next section.
2. LR too high for a tiny dataset → halve it (`--set optim.lr=5e-5`).
3. Bad samples (empty/format-broken) → `uv run rocm-qlora data` validates
   the dataset loudly before any GPU work.

## Mid-run: "grad_norm is NaN" abort (finite loss, poisoned backward)

**Symptom** (observed in a real 27B run): a step reports a normal-looking
loss (e.g. 1.07) but `grad_norm: NaN`; every following step then reports
`loss 0.0` *exactly* while the loss EMA keeps "improving". The run is dead —
the optimizer already wrote NaN into the weights (TRL's fused loss returns
exactly zero once the model is poisoned, because ignored positions contribute
zeros to the sum).

**What the code does now**: the monitor hard-stops on the first of the three
numeric failure modes (`health_violation` in `src/rocm_qlora/monitor.py`:
non-finite loss, NaN grad_norm with finite loss, exact-zero loss), writes an
`error` event into `metrics.jsonl`, and exits loudly. Nothing silent, nothing
saved that looks healthy.

**What to do**:

1. Resume from the last good epoch checkpoint — optimizer/scheduler state is
   included (`save_only_model=False`):

   ```bash
   uv run rocm-qlora train --config <cfg> --resume .outputs/runs/<run-dir>
   ```

2. To hunt the cause, re-run with the env-gated diagnostics:

   ```bash
   ROCM_QLORA_DETECT_ANOMALY=1 ROCM_QLORA_GRAD_CHECK=1 uv run rocm-qlora train …
   ```

   `DETECT_ANOMALY` raises at the exact op whose backward produced the NaN
   (with a traceback); `GRAD_CHECK` names the exact parameter in a hook,
   before the optimizer step.

**Known occurrence & status (open)**: the 27B path shows this intermittently
(~2 events in ~15 observed steps, across both transports and batch configs);
the toy model shows it ~20× more rarely (1 event in ~230 steps). Plain
allreduce is bit-exact on both transports at the 27B's gradient sizes
(`scripts/poc_p2p_allreduce.py`), so the cause is model-side numerics, not
communication. Until it is root-caused, the operational answer is:
guards (already active), `--resume` from the last epoch checkpoint, and — to
hunt it — the env-gated diagnostics above. `scripts/poc_deltanet_backward.py`
isolates one Gated DeltaNet layer at the 27B's exact dimensions for a
targeted reproduction (needs a free GPU).

## "mat1 and mat2 shapes cannot be multiplied" (weird 1×N weights)

This is a **quantization-config mismatch**: a module stored in 16-bit was
constructed as 4-bit (or vice versa). Two known traps, both fixed here:

* TRL 1.16's fused loss slices the raw `lm_head` weight (for every
  `loss_type`) → `lm_head` must stay 16-bit (upstream checkpoints agree).
* At load time the **stored** `quantization_config` must match reality. The
  `prepare` step derives its skip list from the actual modules; don't
  hand-edit it.

If `prepare` logic was edited: rebuild with `--force` and re-run `data`-style
validation (`train --dry-run` catches this class in 2 minutes).

## "unexpected keyword argument 'warmup_ratio'"

transformers v5 removed `warmup_ratio` from training args. This project
computes `warmup_steps` from the ratio internally — keep using
`optim.warmup_ratio` in the TOML.

## OOM on the 27B (24 GB cards)

In order of effectiveness:

1. `--set data.max_length=1280` (or 1024) — activation memory is linear in it.
2. `--set optim.optimizer=adamw_8bit` (default already) or `adafactor`.
3. Confirm `prepare` ran (weights ~19 GB); an un-prepared model needs ~25 GB.
4. Stop other VRAM consumers: `rocm-smi --showpids`. An inference server
   holding most of the VRAM will block training entirely — the preflight
   lists exactly who holds what.

## Training never starts / hangs before step 1

* First `trainer.train()` step includes Triton kernel warmup on this stack —
  give it ~30 s.
* If it hangs >2 min: check `logs/rank-0.log`; a barrier deadlock usually
  means one rank died earlier (look at all rank logs, not just 0).
* HF Hub throttling during model load: `export HF_HUB_DISABLE_XET=1` (known
  AMD-network workaround from the Unsloth AMD guide).

## Slow steps

Baseline expectation: a few seconds/step for 27B QLoRA at batch 2 × 1536
tokens across the GPUs; the toy is sub-second. If steps slow down mid-run:

* check `gpu_health.png` for thermal throttling,
* check `vram.png` for a leak (monotonic climb),
* `torch_empty_cache_steps=50` is already set; fragmentation otherwise is
  rare with `garbage_collection_threshold:0.8`.

## The run "succeeded" but the model is bad

Checklist, in order:

1. `chat` with `/think off` — is the *format* right? (template problem, not
   training).
2. Compare base vs adapter on 3–5 questions with known answers.
3. Look at `report.md`: did eval loss actually improve?
4. A few hundred examples × 3 epochs is a *style/format* fine-tune. If new
   knowledge was the goal, no hyperparameter will deliver it.

## Diagnostics to collect before reporting an issue

```bash
uv run rocm-qlora doctor              # full probe → .context/doctor-latest.json
<run>/env_fingerprint.json            # exact stack + GPU state at run start
<run>/config.resolved.toml            # exact config (overrides applied)
<run>/logs/rank-0.log                 # first ERROR line
```
