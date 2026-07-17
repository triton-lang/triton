"""Correctness tests for the GQA Flash-Attention kernels.

Runs on GPU by default. To run on CPU (no GPU required) use Triton's
interpreter:

    TRITON_INTERPRET=1 pytest test_flash_attn.py -v

The reference is a plain PyTorch attention; we check both the forward output
and the analytic gradients (dq, dk, dv) obtained via autograd.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn import flash_attention  # noqa: E402


def _reference(q, k, v, causal, scale):
    Z, H, N, D = q.shape
    H_KV = k.shape[1]
    group = H // H_KV
    kk = k.repeat_interleave(group, dim=1)
    vv = v.repeat_interleave(group, dim=1)
    s = (q @ kk.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ vv


# (H, H_KV): MHA, MQA, GQA
CONFIGS = [(2, 2), (4, 1), (4, 2), (8, 2)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("H,H_KV", CONFIGS)
def test_forward(causal, H, H_KV):
    torch.manual_seed(0)
    Z, N, D = 1, 32, 16
    q = torch.randn(Z, H, N, D)
    k = torch.randn(Z, H_KV, N, D)
    v = torch.randn(Z, H_KV, N, D)
    scale = 1.0 / math.sqrt(D)
    out = flash_attention(q, k, v, causal=causal, sm_scale=scale, block_m=16, block_n=16)
    ref = _reference(q, k, v, causal, scale)
    assert torch.allclose(out.float(), ref, atol=1e-4, rtol=0), \
        (out - ref).abs().max()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("H,H_KV", CONFIGS)
def test_backward(causal, H, H_KV):
    torch.manual_seed(0)
    Z, N, D = 1, 32, 16
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(Z, H, N, D, requires_grad=True)
    k = torch.randn(Z, H_KV, N, D, requires_grad=True)
    v = torch.randn(Z, H_KV, N, D, requires_grad=True)
    q2 = q.detach().clone().requires_grad_()
    k2 = k.detach().clone().requires_grad_()
    v2 = v.detach().clone().requires_grad_()

    out = flash_attention(q, k, v, causal=causal, sm_scale=scale, block_m=16, block_n=16)
    ref = _reference(q2, k2, v2, causal, scale)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)

    assert torch.allclose(q.grad, q2.grad, atol=1e-3), (q.grad - q2.grad).abs().max()
    assert torch.allclose(k.grad, k2.grad, atol=1e-3), (k.grad - k2.grad).abs().max()
    assert torch.allclose(v.grad, v2.grad, atol=1e-3), (v.grad - v2.grad).abs().max()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
