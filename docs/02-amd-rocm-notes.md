# 02 · AMD ROCm notes (gfx1100 / RDNA3, October 2026)

Everything here was verified empirically on **gfx1100 (RDNA3) hardware,
ROCm 7.1 user-space, torch 2.13.0+rocm7.1, a mainline kernel** — not copied
from generic guides. `rocm-qlora doctor` reproduces the probes.

## Attention: SDPA + AOTriton (works!)

| backend | with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | without |
|---|---|---|
| FLASH | ✅ | ✅ |
| EFFICIENT (mem-efficient) | ❌ `RuntimeError: No available kernel` | ❌ |
| MATH | ✅ | ✅ |

* There is no `flash-attn` package for RDNA3; the working fast path is
  PyTorch **SDPA**, which dispatches to **AOTriton** flash kernels here.
* `doctor` runs each backend in a **fresh subprocess** — on RDNA3 a failed
  kernel launch can be *deferred* and surface at the next checked op,
  contaminating the process and producing misleading errors
  (pytorch#194498 documents exactly this on gfx1100).
* The launcher reads the doctor verdict and sets the env + attention
  implementation automatically. `eager` (math attention) is the safe fallback
  if a future driver/wheel combination breaks the fused path.
* The probes cover **forward** kernels. The one fused kernel observed
  producing non-finite gradients is the *backward* of the mem-efficient path
  (`ScaledDotProductEfficientAttentionBackward0`), and an explicit attention
  mask — which padded batches always pass — is what dispatches to it. The
  launcher therefore **disables every backend the probe flagged** for the
  whole training process: on this machine masked batches run math attention,
  while causal unmasked ones stay on flash. `--set model.attn=eager` forces
  math everywhere; see the NaN note in docs/05.

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
`doctor` probes `hipIpcGetMemHandle` on every run, so transport regressions
are caught immediately.

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

## The 24 GB budget for a 27B model

The math that makes 27B QLoRA fit on 24 GB cards (per GPU, DDP). The
"why" column carries measured reality. The tool does **not** predict this
number: the preflight reports free VRAM as a fact, and a run either fits or
OOMs — this table (plus the knobs in docs/05) is what to react with:

| item | GB | why |
|---|---|---|
| NF4 weights (text-side Linears) | ~12.5 | 24.33 B params at 0.5 B/param packed + 0.4 absmax — measured on disk. `in_proj_a/b` stay 16-bit (48 MB; upstream keeps the gate projections precise) |
| embeddings (bf16, untied) | ~2.5 | `embed_tokens` is not a Linear |
| lm_head (bf16) | ~2.5 | stays 16-bit — TRL's fused loss slices the raw weight; upcasting to fp32 costs +2.5 GB for nothing |
| vision tower | ~0.2 | dropped right after load (freed) |
| LoRA r=16 (all text-side linears) + 8-bit Adam | ~0.7 | ~117 M trainable (0.4 %) |
| activations @1536 tok, batch 1, grad-ckpt | ~1.3 | recomputed, not stored |
| context / fragmentation / RCCL | ~1 | HIP context + transport buffers |
| **total** | **~21 of 24** | measured 21.1 GB at batch 1, step 1 (2026-10-23 run) |

The single most important number: **requantizing `in_proj_qkv` frees ~3.7 GB**
per GPU (5 GB in 16-bit → 1.3 GB at NF4; upstream checkpoints keep it in
16-bit). That's `prepare`'s job.

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
| `NCCL_IB_DISABLE` / `NCCL_DEBUG` | `1` / `WARN` | no InfiniBand probing on these machines; keeps NCCL's startup chatter out of the log |
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
