import pytest
import torch

import triton
import triton.language as tl

cvt = {
    'bool': torch.bool,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

int_dtypes = ['int8', 'int16', 'int32', 'int64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + float_dtypes


@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False)
    for dtype_x in dtypes
    for dtype_z in dtypes
])
def test_fptrunc(dtype_x, dtype_z, bitcast, device='cuda'):
    SIZE = 256
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, **meta):
        off = tl.arange(0, meta['SIZE'])
        x = tl.load(X + off)
        tl.store(Z + off, x)
    # inputs
    x = triton.testing.random(SIZE, dtype=cvt[dtype_x], device=device)

    # reference result
    z_ref = x.type(dtype=cvt[dtype_z])

    # triton result
    z_tri = torch.zeros_like(x, dtype=cvt[dtype_z])

    # triton.testing.assert_almost_equal(z_ref, z_tri)

    print("before kernel")
    # run load and store kernel
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=1)
    print("after kernel")

    # print("x:", x)
    # print("z_ref:", z_ref)
    # print("z_tri:", z_tri)
    # compare
    print("before compare")
    triton.testing.assert_almost_equal(z_ref, z_tri)
    print("after compare")


if __name__ == '__main__':
    test_fptrunc()
