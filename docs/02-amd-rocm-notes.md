# 02 · AMD ROCm notes (gfx1100 / RDNA3, October 2026)

Everything here was verified empirically on **gfx1100 (RDNA3) hardware,
ROCm 7.1 user-space, torch 2.13.0+rocm7.1, a mainline kernel** — not copied
from generic guides. Every decision it implies is a config knob; nothing is
probed at run time.

## Attention: what SDPA does here, and the two knobs

**The concept.** With `model.attn = "sdpa"` (the default) torch picks one of
three kernels *per batch*: **flash** (fused, fastest), **mem-efficient**
(fused, memory-oriented) or **math** (plain, always correct). With
`model.attn = "eager"` everything runs math — slowest, and a safe fallback if
a driver or wheel change breaks the fused paths.

**What was measured on this box.** There is no `flash-attn` package for
RDNA3; the working fast path is SDPA's flash kernel (AOTriton-backed). The
mem-efficient path **fails to launch** here (`RuntimeError: No available
kernel`) and — more importantly — its *backward* was the kernel observed
producing non-finite gradients in real runs, on padded batches (which always
pass an explicit attention mask, the thing that dispatches to it).

**The two knobs, both static:**

* `env.aotriton = true` exports `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  to the workers. Some gfx1100 wheels only expose the fused flash path with
  it; on this box flash works either way. *What to check:* if a stack update
  makes training fail or misbehave in attention, set it `false` and try
  again — it is one env var, not a code path.
* `model.sdpa_disable = ["mem_efficient"]` (what the shipped configs set)
  switches the named backends off in every worker. *What to check:* the
  symptom it prevents is NaN gradients early in a run with padded batches
  (docs/05's NaN section). If you are confident your stack is fine, clear
  the list and watch for that symptom; keeping it costs only the kernel that
  fails to launch here anyway.

`--set model.attn=eager` forces math everywhere; see the NaN note in docs/05.

*What you will see:* on this stack transformers prints once-per-process
notices that `causal_conv1d` / `flash-linear-attention` are not installed and
that it falls back to reference implementations (correct, just slower — the
DeltaNet layers are the ones affected). Training runs write those to
`logs/rank-*.log` rather than the console; `chat` does not show them at all.

## Multi-GPU: RCCL P2P works (dma-buf IPC) — but SHM is the default

Topology (`rocm-smi --showtopo`): consumer gfx1100 cards are
**PCIE**-attached — no XGMI, no PCIe switch.

Out of the box, RCCL's P2P transport died at the first barrier with
`hipIpcGetMemHandle failed : invalid argument`. **Root cause (verified):**
the ROCm 7.1 runtime defaults to *legacy* IPC mode, whose handle-export
ioctl (`AMDKFD_IOC_IPC_EXPORT_HANDLE`) exists only in the out-of-tree DKMS
driver; the in-tree amdgpu driver on mainline kernels does not implement it.
The fix (kept in the launcher):

```
HSA_ENABLE_IPC_MODE_LEGACY=0     # select the mainline dma-buf IPC path
```

Measured multi-GPU: **~26 GB/s bus bandwidth** (vs ~5.2 GB/s SHM), and
`scripts/poc_p2p_allreduce.py` verifies P2P allreduce is **bit-exact** at the
27B's real gradient sizes (30 iterations × 4 sizes, zero mismatches).

**Why the default is nevertheless SHM** (`NCCL_P2P_DISABLE=1`): the 27B
showed rare NaN-gradient events, and they are **model-side, not
transport-side** — they occurred over both transports, and plain allreduce is
bit-exact on both. SHM is a simplicity choice (~45 ms/step for LoRA — noise
next to a 10–25 s step); the NaN guards abort loudly either way. Opt back in:

```toml
[env.extra]
NCCL_P2P_DISABLE = "0"
```

`HSA_FORCE_FINE_GRAIN_PCIE=1` adds nothing here. Do **not** add `iommu=pt`
(translated IOMMU works; one upstream report shows it corrupting peer IPC).
RCCL prints a WARN about the missing `iommu=pt` at init; it is expected and
harmless here, and the workers run at `NCCL_DEBUG=ERROR`, so you will not
see it.
The launcher's default is the shared-memory transport; opting into P2P is
that one `[env.extra]` line above. *What to check:* if you want to know
whether P2P is actually being used (and bit-exact) on your machine, run the
allreduce POC in `scripts/` by hand — it is a manual check, nothing the tool
does for you.

## bitsandbytes 4-bit: 0.50.2 is good on ROCm

* bnb ≤ 0.49.2 had a 4-bit **GEMV NaN bug at decode shape** (`batch=1,
  seq_len=1`) on every ROCm target — training shapes were unaffected, but
  inference/`chat` produced NaNs. Fixed upstream (bnb PR #1887) and present
  in the 0.50.x PyPI wheels.
* `doctor` runs a smoke test at both training shape and decode shape;
  currently: **✅ no NaN, max abs error 0.18 at decode** (normal NF4 error).

## Hybrid attention (Qwen3.8's Gated DeltaNet)

Qwen3.8 interleaves 48 **Gated DeltaNet** layers with 16 full-attention
layers. The fast kernels for the DeltaNet layers live in
`flash-linear-attention` (Triton-based). This project deliberately **does not
install it** — transformers ships pure-PyTorch fallbacks, which are correct
and memory-frugal. The decision is evidence-based:

* **Correctness**: FLA issue #790 — certain Triton autotune configs of
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (the exact kernel this
  model uses) produce **wrong outputs**; a training run diverged with such a
  config. Whether this machine's autotune picks a safe config is unknowable
  without bit-exact reference tests.
* **Cold start**: FLA issue #804 — fresh Triton caches on ROCm can stall the
  first forward for minutes (82 min autotune reported on a gfx1100).

The honest trade: the pure-torch fallback is slower on the 48
linear-attention layers, so the run is compute-bound — keep the micro-batch
as large as VRAM allows. To experiment anyway: `uv add
flash-linear-attention`, run with a *warm* Triton cache, and validate against
the fallback numerically (same weights, same batch → compare losses) before
trusting a long run.

## Allocator

`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8` is set by default —
GC threshold only. Do **not** enable `expandable_segments:True`: tried on
2026-10-23, it silently corrupted numerics (identical seed/batch, step-1 loss
1.43 → 3.36, NaN grads; docs/05). Headroom is bought with batch size instead.

## The per-GPU budget for a 27B model (24 GiB cards)

The numbers that make 27B QLoRA fit on 24 GiB cards (per GPU, DDP; rocm-smi
reports the card as 25.75 GB, the unit this tool prints). The
"why" column carries measured reality. The tool does **not** predict this
number: the preflight reports free VRAM as a fact, and a run either fits or
OOMs — this table (plus the knobs in docs/05) is what to react with:

| item | GB | how it is known |
|---|---|---|
| NF4 weights (text-side Linears) | 12.5 | measured on disk (24.33 B params × 0.5 B packed + 0.4 absmax). `in_proj_a/b` stay 16-bit (48 MB; upstream keeps the gate projections precise) |
| embeddings (bf16, untied) | 2.5 | exact — 1.271 B params × 2 B |
| lm_head (bf16) | 2.5 | exact — same count; stays 16-bit (TRL's fused loss slices the raw weight) |
| vision tower | 0.25 | measured on disk; dropped right after load (freed) |
| LoRA r=16 (all text-side linears) + 8-bit Adam | 0.7 | exact — 116.7 M trainable × (2 params + 2 grads + 2 optimizer bytes) |
| **everything else** | **~2.7** | the measured step-1 total minus the exact rows above: activations, HIP context, RCCL buffers, allocator slack. A *residual*, not a model — and the total itself grows toward ~25.6 GB over a run (allocator sawtooth, docs/06) |
| **total** | **21.1 of 25.75** | measured at batch 1, step 1 (2026-10-23 run); batch 2 × 1536 peaked at 25.5/25.75 and OOMed (docs/05) |

The single most important number: **requantizing `in_proj_qkv` frees ~3.7 GB**
per GPU (5 GB in 16-bit → 1.3 GB at NF4; upstream checkpoints keep it in
16-bit). That's `prepare`'s job.

**Which GPU runs what.** Training uses `dist.gpus` — the launcher resolves
`"auto"` to every visible card, and warns when a card is shared with another
process. The one-off steps, `prepare` and `chat`, run on the **first** GPU
that list names (`"auto"` → GPU 0): a static, predictable choice, not a scan
for the freest card.

## Environment variables that matter (and what the launcher sets)

| var | value | why |
|---|---|---|
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` | selected ids | pin the GPU set |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `0` | dma-buf IPC → RCCL P2P works (~26 GB/s vs ~5.2) |
| `HSA_USE_SVM` | `0` (27B config) | removes the KFD-eviction → MES reset trigger (docs/05) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | enables AOTriton flash kernels |
| `PYTORCH_HIP_ALLOC_CONF` | `garbage_collection_threshold:0.8` | fragmentation control |
| `OMP_NUM_THREADS` | 8 per rank | plenty for dataloading; avoids host oversubscription |
| `TOKENIZERS_PARALLELISM` | `false` | silences fork warnings |
| `NCCL_P2P_DISABLE` | `1` (SHM default) | simplicity default; both transports are bit-exact — set `"0"` via `[env.extra]` for P2P |
| `NCCL_IB_DISABLE` / `NCCL_DEBUG` | `1` / `ERROR` | no InfiniBand probing; RCCL's WARN level only ever produced the known-benign iommu notice here — raise it via `[env.extra]` when debugging transports |
| `AMDGPU_ASIC_ID_TABLE_PATHS` | `/usr/share/libdrm` | silences libdrm's per-GPU `(null): No such file or directory` at HIP init |
| `PYTHONHASHSEED` | `run.seed` | fixed before the interpreter starts; part of reproducibility |
| `PYTHONUNBUFFERED` / `HF_HUB_DISABLE_TELEMETRY` | `1` / `1` | live log lines; no HF telemetry |

`HSA_OVERRIDE_GFX_VERSION` is **not** needed for gfx1100 (native support);
it exists for unsupported gfx targets only.

## Read more

- ROCm PyTorch compatibility — https://rocm.docs.amd.com/en/latest/compatibility/pytorch-compatibility.html
- RCCL usage tips — https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html
- pytorch#194498 (deferred SDPA launch failures on gfx1100) — https://github.com/pytorch/pytorch/issues/194498
- bnb PR #1887 (ROCm GEMV fix) — https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1887
