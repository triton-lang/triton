"""When triton is lowered to llvm it no longer deals with tensors and
 encodings. At this stage all tensor operations are lowered to
 thread-specific code. During this lowering elementwise ops an
 optimization happens where if N elements of an elementwise op are the
 same they are both computed by the same operation. The code was
 buggy, however, at some point and it assumed that each thread is
 assigned contiguous blocks of elements to be computed.

"""
import pytest
import torch
import numpy as np

import triton.language as tl
from triton import jit

input_dtypes = ["float32", "float64"]
out_dtypes = ["float16", "float32"]


def test_elementwise_dedupe():

    def numpy_kernel():
        mask = np.ones((16, 16), dtype=bool)
        mask[:2, :] = False
        masked = np.where(mask, np.zeros((16, 16)), 1)
        return masked @ np.ones((16, 16))

    @jit
    def kernel(out_ptr):
        # Make a 16x16 mask where `mask[0:2,:] == False`
        mask_qk = tl.full([16, 16], 1, dtype=tl.int1)
        mask_qk &= (tl.arange(0, 16) < 2)[:,None]
        # Force the mask encoding to #mma so it needs convert to #dot.
        x = tl.zeros([16,16], tl.float32)
        dot = tl.dot(x, x)
        # make sure it converts to dot. If the layout is not properly
        # taken into account (8, 0) may not be masked.
        res = tl.where(mask_qk, dot, 1)
        res = tl.dot(res, tl.full(x.shape, 1, x.dtype))
        tl.store(to_uniblock16(out_ptr), res)

    grid = (1,)
    triton_res = torch.zeros((16, 16))
    kernel[grid](triton_res)
    numpy_res = numpy_kernel()

    torch.testing.assert_close(triton_res, numpy_res)
