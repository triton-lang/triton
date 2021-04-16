import torch
import triton
import copy
import pytest
import ast

torch.manual_seed(0)

# convert from string to torch.dtype
# Necessary because doesn't print torch.dtype properly
cvt = {
    'bool': torch.bool,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

int_dtypes = ['int8', 'int16', 'int32', 'int64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + float_dtypes


def patch_kernel(template, test_str):
    kernel = copy.deepcopy(template)
    kernel.src = kernel.src.replace('GENERATE_TEST_HERE', test_str)
    return kernel


# generic test functions
def _test_unary(dtype_x, expr, device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, **meta):
        off = triton.arange(0, meta['SIZE'])
        x = triton.load(X + off)
        z = GENERATE_TEST_HERE
        triton.store(Z + off, z)

    kernel = patch_kernel(kernel, expr)
    # inputs
    x = triton.testing.random(SIZE, dtype=cvt[dtype_x], device=device)
    # reference result
    z_ref = eval(expr)
    # triton result
    z_tri = torch.empty_like(z_ref)
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=4)
    # compare
    triton.testing.assert_allclose(z_ref, z_tri)


def _test_binary(dtype_x, dtype_y, expr, device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, Y, **meta):
        off = triton.arange(0, meta['SIZE'])
        x = triton.load(X + off)
        y = triton.load(Y + off)
        z = GENERATE_TEST_HERE
        triton.store(Z + off, z)

    kernel = patch_kernel(kernel, expr)
    # inputs
    x = triton.testing.random(SIZE, dtype=cvt[dtype_x], device=device)
    y = triton.testing.random(SIZE, dtype=cvt[dtype_y], device=device)
    # reference result
    z_ref = eval(expr)
    # triton result
    z_tri = torch.empty(SIZE, dtype=z_ref.dtype, device=device)
    kernel[(1, )](z_tri, x, y, SIZE=SIZE, num_warps=4)
    # compare
    triton.testing.assert_allclose(z_ref, z_tri)


'''
# ---------------
# test binary ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, expr", [
    (dtype_x, dtype_y, f' x {op} y') \
  for op in ['+', '-', '*', '/', '%'] \
  for dtype_x in dtypes \
  for dtype_y in dtypes
])
def test_bin_op(dtype_x, dtype_y, expr, device='cuda'):
    _test_binary(dtype_x, dtype_y, expr, device=device)


# ---------------
# test bitwise ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, expr", [
    (dtype_x, dtype_y, f' x {op} y') \
  for op in ['&', '|', '^', '>>', '<<'] \
  for dtype_x in dtypes \
  for dtype_y in dtypes
])
def test_bitwise_op(dtype_x, dtype_y, expr, device='cuda'):
    if 'float' in dtype_x + dtype_y:
        with pytest.raises(RuntimeError):
            _test_binary(dtype_x, dtype_y, expr, device=device)
    else:
        _test_binary(dtype_x, dtype_y, expr, device=device)


# ---------------
# test compare ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, expr", [
    (dtype_x, dtype_y, f' x {op} y') \
    for op in ['==', '!=', '>', '<', '>=', '<='] \
    for dtype_x in dtypes \
    for dtype_y in dtypes
])
def test_compare_op(dtype_x, dtype_y, expr, device='cuda'):
    _test_binary(dtype_x, dtype_y, expr, device=device)


# ---------------
# test unary ops
# ---------------
@pytest.mark.parametrize("dtype_x, expr", [
    (dtype_x, f' -x') for dtype_x in float_dtypes
] + [\
    (dtype_x, f' ~x') for dtype_x in int_dtypes
     ])
def test_unary_op(dtype_x, expr, device='cuda'):
    _test_unary(dtype_x, expr, device=device)
'''

# ----------------
# test indexing
# ----------------

_all = slice(None, None, None)

smap = {'None': None, ':': slice(None, None, None)}


@pytest.mark.parametrize("slices", [':,None'])
def test_index1d(slices, device='cuda'):
    slices = tuple([smap[s] for s in slices.split(',')])

    @triton.jit
    def kernel(Z, X, **meta):
        SIZE = meta['SIZE']
        m = triton.arange(0, SIZE)
        n = triton.arange(0, SIZE)
        x = triton.load(X + m)
        z = x[:, None]
        triton.store(Z + m[:, None] * SIZE + n[None, :], z)

    dtype = torch.float32
    shape_in = (32, )
    shape_ou = (32, 32)
    x = triton.testing.random(shape_in, dtype=dtype, device=device)
    y = torch.zeros(shape_ou, dtype=dtype, device=device)
    z_ref = x[slices] + y
    z_tri = torch.empty_like(z_ref)
    kernel[(1, )](z_tri, x, num_warps=1, SIZE=shape_in[0])
    triton.testing.assert_allclose(z_ref, z_tri)


# ---------------
# test load
# ---------------

# ---------------
# test store
# ---------------

# ---------------
# test if
# ---------------

# ---------------
# test for
# ---------------

# ---------------
# test while
# ---------------
