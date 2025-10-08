# fmt: off

import torch
import triton
import triton.language as tl


@triton.jit
def fast_log2(x):

    return tl.extra.libdevice.fast_log2f(x)


@triton.jit
def fast_log2_inplace(X, n, N : tl.constexpr = 256):

    idxs = tl.arange(0, N)
    mask = idxs < n
    x = tl.load(X + idxs, mask=mask, other=1.0)
    x = fast_log2(x)
    tl.store(X + idxs, x, mask=mask)


def test_libdevice():

    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA target.")
        return

    x = torch.tensor([2.0, 4.0, 8.0], device="cuda", dtype=torch.float32)
    fast_log2_inplace[(1,)](x, x.shape[0])
    y = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)

    assert torch.equal(x, y)
