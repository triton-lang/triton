import torch
from torch.testing import assert_close

import triton
import triton.language as tl


def test_device_assert():
    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.device_assert(x == 0, "x != 0")
        tl.store(Y + tl.arange(0, BLOCK), x)

    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0])
    assert_close(y, x)


if __name__ == "__main__":
    test_device_assert()
