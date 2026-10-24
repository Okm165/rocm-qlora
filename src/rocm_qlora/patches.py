"""Runtime patches: the DDP init-broadcast skip and env-gated diagnostics.

Everything here touches a third-party runtime deliberately, so it lives apart
from the worker's phases:

* ``skip_redundant_init_sync`` — monkeypatches DDP's init broadcast away (a
  ~4.9 GiB transient at 27B; redundant because every rank loads the same
  checkpoint and seeds identically). It warns loudly if torch renames the
  binding.
* ``install_diagnostics`` — env-gated (``ROCM_QLORA_DETECT_ANOMALY``,
  ``ROCM_QLORA_GRAD_CHECK``) numerical checks for the NaN-gradient hunt;
  off by default.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def skip_redundant_init_sync(log: Callable[[str], None]) -> None:
    """Disable DDP's initial parameter broadcast (safe *by construction* here).

    The broadcast coalesces the 16-bit parameters into one ~4.9 GiB transient
    buffer (fp16 embed_tokens + lm_head), which OOMs a 24 GB card next to the
    loaded weights. It is redundant: every rank loads the same prepared
    checkpoint and initializes LoRA with the same seed, so the replicas are
    bit-identical by construction. This is a deliberate monkeypatch (torch
    exposes no parameter for it, and accelerate's kwargs dataclass cannot
    carry one); if a future torch renames the binding, the warning below says
    the broadcast — and its OOM — is back.
    """
    import torch.nn.parallel.distributed as ddp_mod

    if hasattr(ddp_mod, "_sync_module_states"):
        ddp_mod._sync_module_states = lambda *args, **kwargs: None  # pyright: ignore[reportPrivateUsage]  # deliberate no-op
        log("DDP init broadcast disabled (identical checkpoints + shared seed → replicas identical by construction)")
    else:
        log(
            "WARNING: torch.nn.parallel.distributed._sync_module_states is gone — the DDP init broadcast "
            "(~4.9 GiB transient at 27B) is back; expect an OOM at trainer construction"
        )


def _install_grad_finite_checks(model: torch.nn.Module, log: Callable[[str], None]) -> None:
    """Abort at the *exact* parameter whose gradient goes non-finite.

    The failure this exists for: a step with a finite loss can still produce
    NaN gradients in the backward. DDP allreduces the NaN into every rank,
    the optimizer step poisons the weights, and from then on the fused loss
    reports exactly 0.0 while the loss EMA keeps falling. Catching the NaN in
    the post-accumulate hook fires *before* the optimizer step and names the
    tensor — the difference between "grad_norm became NaN" and a root cause.

    Diagnostics only (env-gated): one isfinite kernel + sync per trainable
    tensor per micro-batch.
    """
    import torch

    def make_hook(param_name: str):
        def hook(grad: torch.Tensor):
            if grad is not None and not bool(torch.isfinite(grad).all()):  # pyright: ignore[reportUnnecessaryComparison]  # defensive
                raise RuntimeError(
                    f"non-finite gradient in '{param_name}' — aborting before the optimizer step "
                    "(ROCM_QLORA_GRAD_CHECK=1). See docs/05-troubleshooting.md"
                )

        return hook

    n = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_post_accumulate_grad_hook(make_hook(name))
            n += 1
    log(f"gradient finiteness checks installed on {n} trainable tensors")


def install_diagnostics(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Env-gated numerical diagnostics (see ``metrics.health_violation`` for the failure mode)."""
    if os.environ.get("ROCM_QLORA_DETECT_ANOMALY") == "1":
        import torch

        torch.autograd.set_detect_anomaly(True)
        logger.info("autograd anomaly detection ON — the first NaN-producing op will raise with a traceback")
    if os.environ.get("ROCM_QLORA_GRAD_CHECK") == "1":
        _install_grad_finite_checks(model, logger.info)
