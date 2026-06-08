import pytest
import torch

import triton
import triton.language as tl


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


@triton.jit
def _f32x2_kernel(a_ptr, b_ptr, c_ptr, out_mul, out_add, out_sub, out_fma, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(out_mul + offs, tl.math.mul_f32x2(a, b))
    tl.store(out_add + offs, tl.math.add_f32x2(a, b))
    tl.store(out_sub + offs, tl.math.sub_f32x2(a, b))
    tl.store(out_fma + offs, tl.math.fma_f32x2(a, b, c))


@pytest.mark.skipif(not _is_blackwell(), reason="f32x2 requires Blackwell sm_100")
def test_f32x2():
    N = 256
    torch.manual_seed(0)
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.randn(N, device="cuda", dtype=torch.float32)
    out_mul = torch.empty_like(a)
    out_add = torch.empty_like(a)
    out_sub = torch.empty_like(a)
    out_fma = torch.empty_like(a)
    _f32x2_kernel[(1, )](a, b, c, out_mul, out_add, out_sub, out_fma, N)
    torch.testing.assert_close(out_mul, a * b)
    torch.testing.assert_close(out_add, a + b)
    torch.testing.assert_close(out_sub, a - b)
    torch.testing.assert_close(out_fma, a * b + c)
