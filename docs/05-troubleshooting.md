# 05 · Troubleshooting (real failures, real fixes)

Every entry below is a real failure or finding from building this project on
gfx1100 / ROCm 7.1 — not generic advice.

## Where to look first

```
<run>/logs/rank-0.log      the full story (tracebacks, warnings)
<run>/metrics.jsonl        the numeric story (events have type="event")
                           schema: the MetricsRecord TypedDicts in metrics.py
                           (train / eval / epoch_end / event / run_end)
.context/doctor-latest.json  machine state at run start
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

**Ignore RCCL's `Missing "iommu=pt" from kernel command line` warning** at
launch — it is printed unconditionally by this RCCL build and the advice is
wrong for this stack (see above).

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
numeric failure modes (`health_violation` in `src/rocm_qlora/metrics.py`:
non-finite loss, NaN grad_norm with finite loss, exact-zero loss), writes an
`error` event into `metrics.jsonl`, and exits loudly. Nothing silent, nothing
saved that looks healthy.

**What to do**:

1. Resume from the last good checkpoint (27B config: every 42 steps;
   optimizer/scheduler state included — `save_only_model=False`):

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
the toy shows it rarely (3 events in ~330 steps on `sdpa`; 0 in 138 steps with
math attention). `ROCM_QLORA_DETECT_ANOMALY=1` pinned the first non-finite
backward op to **`ScaledDotProductEfficientAttentionBackward0`** — the fused
(mem-efficient) attention backward, not the DeltaNet layers; plain allreduce
is bit-exact on both transports, so the cause is model-side numerics. Until
it is root-caused, the operational answer is: guards (already active),
`--resume` from the last checkpoint, and the automatic backend guard (the
launcher disables probe-flagged backends, so the mem-efficient backward is
never dispatched to; `--set model.attn=eager` forces math everywhere).
`scripts/poc_deltanet_backward.py` isolates one Gated DeltaNet layer at the
27B's exact dimensions for a targeted reproduction (needs a free GPU).

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
3. Confirm `prepare` ran (weights ~17.7 GB); an un-prepared model needs ~25 GB.
4. Stop other VRAM consumers: `rocm-smi --showpids`. An inference server
   holding most of the VRAM will block training entirely — the preflight
   lists exactly who holds what.
5. **First choice on this box (2026-10-23): `per_device_batch_size = 1` with
   `grad_accum = 4`** — same effective batch, ~half the activation memory.
   At batch 2 × 1536 the measured peaks were 25.5/25.75 GB and the run OOMed
   at step 25 with 8 near-misses before that.

**Do not enable `expandable_segments`.** It was tried on 2026-10-23 and
silently corrupted numerics on this stack (torch 2.13+rocm7.1/gfx1100):
identical seed/batch/tokens_seen, but step-1 loss 1.43 → 3.36 and grad_norm
NaN. If you ever see a *different* loss on the same data/seed after touching
allocator config, suspect corruption, not the data.

## Training never starts / hangs before step 1

* The first `trainer.train()` step includes kernel warmup on this stack —
  measured ~19 s (later steps ~16–23 s at 27B, batch 1 × 1536).
* If it hangs >2 min: check `logs/rank-0.log`; a barrier deadlock usually
  means one rank died earlier (look at all rank logs, not just 0).
* HF Hub throttling during model load: `export HF_HUB_DISABLE_XET=1` (known
  AMD-network workaround from the Unsloth AMD guide).

## Slow steps

Baseline expectation: a few seconds/step for 27B QLoRA at batch 1 × 1536
tokens across the GPUs; the toy is sub-second. If steps slow down mid-run:

* check `gpu_health.png` for thermal throttling,
* check `vram.png` for a leak (monotonic climb),
* `torch_empty_cache_steps=50` is already set; fragmentation otherwise is
  rare with `garbage_collection_threshold:0.8`.

## GPU reset: "MES failed to respond" → mode1 reset → all ranks die (SIGABRT)

Seen 2026-10-23 on the 27B run (step 4, RX 7900 XTX / gfx1100). Signature,
in order:

1. training stalls mid-step (GPUs spin at 100%; no new `metrics.jsonl` lines),
2. `HW Exception by GPU node-N … reason :GPU Hang` on the console,
3. kernel log: `MES failed to respond to msg=REMOVE_QUEUE` → `MES might be in
   unrecoverable state` → `Failed to evict queue` → `GPU reset begin!. Source: 3`
   → `MODE1 reset` → `VRAM is lost due to GPU reset!`,
4. the process aborts (SIGABRT, exit 134) — all ranks die with it.

What it is: a firmware-level wedge of the GPU's MES (queue scheduler). The
driver can no longer tear the hung queues down and mode1-resets the card;
VRAM is lost, so every process using that card dies. It is **not** our code,
not RCCL, and not the NaN issue above.

Why this machine was vulnerable (measured on this box):

* the installed `gc_11_0_0_mes_2.bin` firmware blob is byte-identical to the
  revision AMD reverted upstream on 2026-09-18 ("causes instability",
  freedesktop work-item 5711) — `linux-firmware-20260916` predates the revert;
* the card's PCIe link (root port `40:03.1`) emits a chronic storm of
  *corrected* replay-timeout errors (`[12] Timeout`, Data Link Layer) under
  GPU load — a link-quality anomaly, plausible trigger/aggravator;
* the same failure class has a documented KFD trigger: an SVM invalidation /
  memory-pressure event can push KFD into queue eviction while queues are
  busy, and MES fails `REMOVE_QUEUE` (ROCm issue #6386).

Mitigations (in place):

* `HSA_USE_SVM=0` in `[env.extra]` of `configs/qwen3.8-27b.toml` — removes
  the documented eviction trigger; ~0.1% measured cost, outputs unchanged;
* mid-epoch checkpoints (`logging.checkpoint_every_n_steps = 42`) — a reset
  now loses ≤ ~15 min (42 steps × ~21 s/step measured); resume with `--resume`;
* keep VRAM pressure low (see "OOM on the 27B" above) — memory pressure is
  what feeds the eviction path.

Fix at the source (operator action): install `linux-firmware` newer than the
2026-09-18 reverts (`sudo dnf update linux-firmware`, then reboot) so the
restored MES blob is loaded. Check which blob is installed:

```bash
xz -dc /lib/firmware/amdgpu/gc_11_0_0_mes_2.bin.xz | sha256sum
# reverted/known-bad: 6e1926d197d9e5f17830c1bba661ac170e2c9e7aba1da56ef26c6f6978b3b79e
# restored/good:      90352e532f3202fa2bc08a88389d47e6627bf2cdcf02080370c22f8aaa198824
```

If it recurs: capture the GPU coredump before it is consumed —
`sudo cat /sys/class/drm/cardN/device/devcoredump/data > gpu-coredump.bin`
(the file disappears on its own within minutes).

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

For HF/TRL internals in the rank logs: `ROCM_QLORA_HF_LOG_LEVEL=info`.
`ROCM_QLORA_LOG_LEVEL` (default `INFO`) sets the package's own file-log level —
`WARNING` quiets the per-step health lines.
