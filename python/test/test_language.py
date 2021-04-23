import torch
import triton
import triton.language as tl
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


def patch_kernel(template, to_replace):
    kernel = copy.deepcopy(template)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


# generic test functions
def _test_unary(dtype_x, expr, device='cuda'):
    SIZE = 128
    # define the kernel / launch-grid
    @triton.jit
    def kernel(Z, X, **meta):
        off = tl.arange(0, meta['SIZE'])
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
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
        off = tl.arange(0, meta['SIZE'])
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
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
  for op in ['&', '|', '^'] \
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


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


@pytest.mark.parametrize("expr", [f'x[{s}]' for s in
    ['None, :', ':, None',\
     'None, :, :', ':, :, None']\
])
def test_index1d(expr, device='cuda'):
    dtype = torch.int32
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, **meta):
        SIZE = meta['SIZE']
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    to_replace = {
        'X_PTR_EXPR': make_ptr_str('X', shape_x),
        'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
        'GENERATE_TEST_HERE': expr,
    }
    kernel = patch_kernel(kernel, to_replace)

    # torch result
    x = triton.testing.random(shape_x, dtype=dtype, device=device)
    y = torch.zeros(shape_z, dtype=dtype, device=device)
    z_ref = eval(expr) + y
    # triton result
    z_tri = torch.empty_like(z_ref)
    kernel[(1, )](z_tri, x, num_warps=1, SIZE=shape_x[0])
    # compare
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
