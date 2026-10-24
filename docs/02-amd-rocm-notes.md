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

Out of the box, RCCL's P2P transport died at:

```
NCCL WARN hipIpcGetMemHandle failed : invalid argument
→ ncclUnhandledCudaError: Call to CUDA function failed  (at the first barrier)
```

**Root cause (verified):** the ROCm 7.1 runtime bundled with the torch wheel
defaults to *legacy* IPC mode. Legacy mode exports IPC handles via
`AMDKFD_IOC_IPC_EXPORT_HANDLE` — an ioctl that only exists in the out-of-tree
DKMS driver; the in-tree amdgpu driver on mainline kernels does not implement
it, so the handle export fails and the P2P transport dies.

**The fix that makes P2P work** (kept in the launcher):

```
HSA_ENABLE_IPC_MODE_LEGACY=0     # select the mainline dma-buf IPC path
```

Measured multi-GPU: **~26 GB/s bus bandwidth** (vs ~5.2 GB/s SHM), and
`scripts/poc_p2p_allreduce.py` verifies P2P allreduce is **bit-exact** at the
27B's real gradient sizes (30 iterations × 4 sizes, zero mismatches).

**Why the default is nevertheless SHM** (`NCCL_P2P_DISABLE=1`): the 27B
training path showed rare, stochastic NaN-gradient events (a finite loss with
`grad_norm = NaN`, after which the optimizer wrote NaN into the weights and
the fused loss reported exactly 0.0 forever). **Those events are model-side,
not transport-side** — they occurred over both P2P and SHM, and plain
allreduce is bit-exact on both (100 iterations × 4 sizes, zero mismatches, at
the 27B's exact gradient sizes). The SHM default is therefore a simplicity
choice — it keeps the IPC/IOMMU path out of the picture, and for LoRA the
cost is ~45 ms/step, noise next to a ~10–25 s step. The NaN guards
(`docs/05`) abort loudly on either transport.

Opt back into P2P for an experiment:

```toml
[env.extra]
NCCL_P2P_DISABLE = "0"
```

`HSA_FORCE_FINE_GRAIN_PCIE=1` adds nothing here (25.8 vs 26.0 GB/s). Do **not**
add `iommu=pt` (translated IOMMU works; one upstream report shows `iommu=pt`
corrupting peer IPC). RCCL logs a "Missing iommu=pt" warning — expected here;
our P2P is verified without it, so revisit only if P2P failures appear.
`rocm-qlora doctor` probes `hipIpcGetMemHandle` on every run so transport
regressions are caught immediately.

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
install it** — transformers ships pure-PyTorch fallbacks
(`torch_chunk_gated_delta_rule`, `causal_conv1d_fn`), which are correct and
memory-frugal. The decision is evidence-based, not conservatism:

* **Correctness**: FLA issue #790 — certain Triton autotune configs
  (`num_warps > num_stages > 1`) of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
  (the exact kernel this model uses) produce **wrong outputs**; a Qwen3.5
  training run diverged after a few hundred steps with such a config.
  Whether the config Triton picks on *this* machine is safe is unknowable
  without bit-exact reference tests — not a trade worth making silently in a
  training run.
* **Cold start**: FLA issue #804 — fresh Triton caches on ROCm can stall the
  first forward for minutes (measured autotune: 82 minutes on a gfx1100 in a
  community report). Weights/VRAM are not the constraint; time-to-first-step
  is.

The honest trade: the pure-torch fallback is slower on the 48 linear-attention
layers, so the run is compute-bound — keep the micro-batch as large as VRAM
allows (the 27B config uses batch 1 × accum 4 after the 2026-10-23 OOMs) to maximize arithmetic intensity. That is
the price of correctness on RDNA3 today. To experiment anyway:
`uv add flash-linear-attention`, run with a *warm* Triton cache, and validate
against the fallback numerically (same weights, same batch → compare losses)
before trusting a long run.

## Allocator

`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8` is set by default —
GC threshold only. Do **not** enable `expandable_segments:True`: tried on
2026-10-23, it silently corrupted numerics (identical seed/batch, step-1 loss
1.43 → 3.36, NaN grads; docs/05). Headroom is bought with batch size instead.

## The 24 GB budget for a 27B model

The math that makes 27B QLoRA fit on 24 GB cards (per GPU, DDP). The
"why" column carries measured reality:

| item | GB | why |
|---|---|---|
| NF4 weights (fully quantized text side) | ~12.8 | ~24.4 B params at 0.5 B/param packed (+absmax) — measured on disk |
| embeddings (bf16, untied) | ~2.5 | `embed_tokens` is not a Linear |
| lm_head (bf16) | ~2.5 | stays 16-bit — TRL's fused loss slices the raw weight; upcasting to fp32 costs +2.5 GB for nothing |
| vision tower | ~0.2 | dropped right after load (freed) |
| LoRA r=16 all-linear + 8-bit Adam | ~0.7 | ~117 M trainable (0.4 %) |
| activations @1536 tok, batch 1, grad-ckpt | ~1.3 | recomputed, not stored |
| context / fragmentation / RCCL | ~1 | HIP context + transport buffers |
| **total** | **~21 of 24** | measured 21.1 GB at batch 1, step 1 (2026-10-23 run) |

The single most important number: **requantizing `in_proj_qkv` saves ~5 GB**
per GPU (upstream checkpoints keep it in 16-bit). That's `prepare`'s job.

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

`HSA_OVERRIDE_GFX_VERSION` is **not** needed for gfx1100 (native support);
it exists for unsupported gfx targets only.

## Read more

- ROCm PyTorch compatibility — https://rocm.docs.amd.com/en/latest/compatibility/pytorch-compatibility.html
- RCCL usage tips — https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html
- pytorch#194498 (deferred SDPA launch failures on gfx1100) — https://github.com/pytorch/pytorch/issues/194498
- bnb PR #1887 (ROCm GEMV fix) — https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1887
