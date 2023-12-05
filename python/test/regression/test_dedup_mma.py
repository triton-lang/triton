"""When triton is lowered to llvm it no longer deals with tensors and
 encodings. At this stage all tensor operations are lowered to
 thread-specific code. During this lowering elementwise ops an
 optimization happens where if N elements of an elementwise op are the
 same they are both computed by the same operation. The code was
 buggy, however, at some point and it assumed that each thread is
 assigned contiguous blocks of elements to be computed.

"""

import os
import pytest
import torch
import numpy as np

import triton.language as tl
from triton import jit


def test_elementwise_dedupe():
    # XXX: Could probably be achieved with interpreter mode aswell.
    def numpy_kernel():
        mask = np.ones((16, 16), dtype=bool)
        mask &= (np.arange(0, 16) < 2)[:, None]
        x = np.zeros((16, 16), dtype=np.float32)
        dot = x @ x
        masked = np.where(mask, dot, 1)
        return masked @ np.ones((16, 16), dtype=np.float32)

    @jit
    def kernel(out_ptr):
        # Make a 16x16 mask where `mask[0:2,:] == False`
        mask = tl.full([16, 16], 1, dtype=tl.int1)
        mask &= (tl.arange(0, 16) < 2)[:, None]
        # Force the mask encoding to #mma so it needs convert to #dot.
        x = tl.zeros([16, 16], tl.float32)
        # make sure it converts to dot
        dot = tl.dot(x, x)
        res = tl.where(mask, dot, 1)
        res = tl.dot(res, tl.full(x.shape, 1, x.dtype))
        out_ptr = tl.make_block_ptr(
            out_ptr,
            shape=(16, 16),
            strides=(16, 1),
            offsets=(0, 0),
            block_shape=(16, 16),
            order=(0, 1),
        )
        tl.store(out_ptr, res)

    baseline_res = numpy_kernel()

    gpu_res = torch.zeros((16, 16))
    kernel[(1,)](gpu_res)

    torch.testing.assert_close(triton_res, baseline_res)
