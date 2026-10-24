"""The deliberate patches: the grad-finiteness diagnostics (CPU-only).

`skip_redundant_init_sync` needs a live DDP group, so it is covered by the
runbook's GPU checks; the hook contract below is pure autograd.
"""

from __future__ import annotations

import pytest
import torch

from rocm_qlora.patches import _install_grad_finite_checks


def test_grad_hook_reads_the_gradient_not_the_parameter():
    """torch calls post-accumulate hooks with the *parameter*.

    Checking the hook argument would test the weights instead of the gradient
    and the diagnostic would never fire (the bug this test pins).
    """
    model = torch.nn.Linear(2, 2, bias=False)
    _install_grad_finite_checks(model, log=lambda _msg: None)

    with pytest.raises(RuntimeError, match="non-finite gradient in 'weight'"):
        (model.weight * float("nan")).sum().backward()


def test_grad_hook_stays_quiet_on_finite_gradients():
    model = torch.nn.Linear(2, 2, bias=False)
    _install_grad_finite_checks(model, log=lambda _msg: None)

    model(torch.ones(1, 2)).sum().backward()  # finite gradients: no error
    assert torch.isfinite(model.weight.grad).all()
