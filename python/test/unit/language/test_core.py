import torch
import triton
import triton.language as tl
import copy
import pytest
import ast
import itertools

torch.manual_seed(0)

# convert from string to torch.dtype
# Necessary because doesn't print torch.dtype properly
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


def patch_kernel(template, to_replace):
    kernel = copy.deepcopy(template)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


@pytest.mark.parametrize("dtype_x", [
    (dtype_x) for dtype_x in dtypes
])
def test_empty_kernel(dtype_x, device='cuda'):
    SIZE = 128
    @triton.jit
    def kernel(X, **meta):
        pass
    x = triton.testing.random(SIZE, dtype=cvt[dtype_x], device=device)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)

# generic test functions
def _test_unary(dtype_x, expr, torch_expr=None, device='cuda'):
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
    if 'log' in expr: x = torch.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if torch_expr is None else torch_expr)
    # triton result
    z_tri = torch.empty_like(z_ref)
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=4)
    # compare
    triton.testing.assert_almost_equal(z_ref, z_tri)


def _test_binary(dtype_x, dtype_y, expr, mode_x='real', mode_y='real', device='cuda'):
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
    if mode_x == 'nan': x[:] = float('nan')
    if mode_y == 'nan': y[:] = float('nan')
    # reference result
    z_ref = eval(expr)
    # triton result
    z_tri = torch.empty(SIZE, dtype=z_ref.dtype, device=device)
    kernel[(1, )](z_tri, x, y, SIZE=SIZE, num_warps=4)
    # compare
    triton.testing.assert_almost_equal(z_ref, z_tri, err_msg=expr)


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
ops = ['==', '!=', '>', '<', '>=', '<=']
@pytest.mark.parametrize("dtype_x, dtype_y, expr, mode_x, mode_y", \
# real
[
    (dtype_x, dtype_y, f' x {op} y', 'real', 'real') \
    for op in ops \
    for dtype_x in dtypes \
    for dtype_y in dtypes
] + \
# NaNs
[('float32', 'float32', f' x {op} y', mode_x, mode_y) \
    for op in ops
    for mode_x, mode_y in [('nan' , 'real'), 
                           ('real', 'nan'), 
                           ('nan' , 'nan')]

])
def test_compare_op(dtype_x, dtype_y, expr, mode_x, mode_y, device='cuda'):
    _test_binary(dtype_x, dtype_y, expr, mode_x=mode_x, mode_y=mode_y, device=device)


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
# test math ops
# ----------------
# @pytest.mark.paramterize("expr", [
#     'exp', 'log', 'cos', 'sin'
# ])

@pytest.mark.parametrize("expr", [
    'exp', 'log', 'cos', 'sin'
])
def test_math_op(expr, device='cuda'):
    _test_unary('float32', f'tl.{expr}(x)', f'torch.{expr}(x) ', device=device)


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
    triton.testing.assert_almost_equal(z_ref, z_tri)


# ---------------
# test tuples
# ---------------


@triton.jit
def fn(a, b):
    return a + b, \
            a - b, \
            a * b


def test_tuples():
    device = 'cuda'

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


# ---------------
# test atomics
# ---------------
@pytest.mark.parametrize("op, dtype_x, mode", itertools.chain.from_iterable([
    [('add', 'int32', mode), ('add', 'float16', mode), ('add', 'float32', mode), \
    ('max', 'int32', mode), ('max', 'float32', mode),\
    ('min', 'int32', mode), ('min', 'float32', mode),\
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x, mode, device='cuda'):
    dtype_x = cvt[dtype_x]
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z, **meta):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x)'})
    torch_op = {'add': torch.sum, 'max': torch.max, 'min': torch.min}[op]
    max_neutral = float('-inf') if dtype_x.is_floating_point else torch.iinfo(dtype_x).min
    min_neutral = float('inf') if dtype_x.is_floating_point else torch.iinfo(dtype_x).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    x_tri = triton.testing.random((n_programs, ), dtype=dtype_x, device=device)
    if mode == 'all_neg':
        x_tri = -torch.abs(x_tri)
    if mode == 'all_pos':
        x_tri = torch.abs(x_tri)
    if mode == 'min_neg':
        idx = torch.randint(n_programs, size=(1, )).item()
        x_tri[idx] = -torch.max(torch.abs(x_tri)) - 1
    if mode == 'max_pos':
        idx = torch.randint(n_programs, size=(1, )).item()
        x_tri[idx] = torch.max(torch.abs(x_tri)) + 1

    z_tri = torch.empty([], dtype=dtype_x, device=device)
    z_tri.fill_(neutral)
    kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = torch_op(x_tri).to(dtype_x)
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == z_tri.item()
    else:
        triton.testing.assert_almost_equal(z_ref, z_tri)


# ---------------
# test cast
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False) \
                        for dtype_x in dtypes\
                        for dtype_z in dtypes
] + [ 
    ('float32', 'bfloat16', False),
    ('bfloat16', 'float32', False),
    ('float32', 'int32', True)
])
def test_cast(dtype_x, dtype_z, bitcast, device='cuda'):
    x = torch.tensor([43.5], dtype=cvt[dtype_x], device=device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, **meta):
        x = tl.load(X)
        z = x.to(Z.dtype.element_ty, bitcast=meta['BITCAST'])
        tl.store(Z, z)

    # triton result
    z_tri = torch.empty((1, ), dtype=cvt[dtype_z], device=device)
    kernel[(1, )](x, z_tri, BITCAST=bitcast)
    # torch result
    if bitcast:
        import numpy as np
        z_ref = x.detach().cpu().numpy().view(getattr(np, dtype_z))
        z_ref = torch.from_numpy(z_ref).to(device)
    else:
        z_ref = x.to(z_tri.dtype)
    assert z_tri == z_ref

# ---------------
# test reduce
# ---------------
@pytest.mark.parametrize("dtype, shape", 
  [(dtype, shape) \
        for dtype in dtypes\
        for shape in [128, 512]])
def test_reduce1d(dtype, shape, device='cuda'):
    dtype = cvt[dtype]

    # triton kernel
    @triton.jit
    def kernel(X, Z, **meta):
        x = tl.load(X + tl.arange(0, meta['BLOCK']))
        tl.store(Z, tl.sum(x, axis=0))

    x = triton.testing.random((shape,), dtype=dtype, device=device)
    # triton result
    z_tri = triton.testing.random((1,), dtype=dtype, device=device)
    kernel[(1,)](x, z_tri, BLOCK=shape)
    # torch result
    z_ref = torch.sum(x).to(dtype)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)


@pytest.mark.parametrize("dtype, shape, axis", 
  [(dtype, shape, 1) \
        for dtype in ['float32']\
        for shape in [(1, 1024)]])
def test_reduce2d(dtype, shape, axis, device='cuda'):
    dtype = cvt[dtype]
    # triton kernel
    @triton.jit
    def kernel(X, Z, **meta):
        range_m = tl.arange(0, meta['BLOCK_M'])
        range_n = tl.arange(0, meta['BLOCK_N'])
        x = tl.load(X + range_m[:, None]*meta['BLOCK_N'] + range_n[None, :])
        z = tl.sum(x, axis=meta['AXIS'])
        tl.store(Z + range_m, z)
    # input
    x = triton.testing.random(shape, dtype=dtype, device=device)
    # triton result
    z_tri = torch.empty((shape[0],), dtype=dtype, device=device)
    kernel[(1,)](x, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis)
    # torch result
    z_ref = torch.sum(x, axis=axis).to(dtype)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)

# ---------------
# test permute
# ---------------

# ---------------
# test permute
# ---------------

@pytest.mark.parametrize("dtype, shape, perm",
  [(dtype, shape, perm) \
        for dtype in ['float32']\
        for shape in [(128, 128)]\
        for perm  in [(1, 0)]])
def test_permute(dtype, shape, perm, device='cuda'):
    dtype = cvt[dtype]
    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn, 
               Z, stride_zm, stride_zn, **meta):
        BLOCK_M = meta['BLOCK_M']
        BLOCK_N = meta['BLOCK_N']
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))
    # input
    x = triton.testing.random(shape, dtype=dtype, device=device)
    # triton result
    z_tri = torch.empty_like(x)
    pgm = kernel[(1, 1)](x, x.stride(0), x.stride(1), 
                        z_tri, z_tri.stride(1), z_tri.stride(0), 
                        BLOCK_M=shape[0], BLOCK_N=shape[1])
    # torch result
    z_ref = x.permute(*perm).contiguous()
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)
    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx

# ---------------
# test dot
# ---------------

@pytest.mark.parametrize("epilogue", ['none', 'add-matrix', 'add-rows', 'add-cols'])
def test_dot(epilogue, device='cuda'):
    torch.manual_seed(0)
    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk, 
               Y, stride_yk, stride_yn,
               Z, stride_zm, stride_zn, **meta):
        BLOCK_M = meta['BLOCK_M']
        BLOCK_K = meta['BLOCK_K']
        BLOCK_N = meta['BLOCK_N']
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        z = tl.dot(tl.load(Xs), tl.load(Ys))
        if meta['ADD_MATRIX']:
            z += tl.load(Zs)
        if meta['ADD_ROWS']:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if meta['ADD_COLS']:
            ZCs = Z + off_n * stride_zn 
            z += tl.load(ZCs)[None, :]
        tl.store(Zs, z)
    # input
    M, N, K = 64, 64, 32
    x = triton.testing.random((M, K), dtype=torch.float16, device=device)
    y = triton.testing.random((K, N), dtype=torch.float16, device=device)
    # triton result
    z = triton.testing.random((M, N), dtype=torch.float16, device=device)
    z_tri = z.clone()
    pgm = kernel[(1, 1)](x, x.stride(0), x.stride(1),
                         y, y.stride(0), y.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX = epilogue=='add-matrix',
                         ADD_ROWS = epilogue=='add-rows',
                         ADD_COLS = epilogue=='add-cols')
    # torch result
    z_ref = torch.matmul(x.float(), y.float())
    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:,0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0,:][None, :]
    z_ref = z_ref.to(torch.float16)
    # compare
    ptx = pgm.asm['ptx']
    # print(ptx)
    triton.testing.assert_almost_equal(z_tri, z_ref)
    # make sure ld/st are vectorized
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx

# ---------------
# test arange
# ---------------

@pytest.mark.parametrize("start", [0, 1, 7, 16])
def test_arange(start, device='cuda'):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)
    @triton.jit
    def _kernel(z, **meta):
        off = tl.arange(0, meta['BLOCK'])
        val = tl.arange(meta['START'], meta['END'])
        tl.store(z + off, val)
    _kernel[(1,)](z_tri, START=start, END=start+BLOCK, BLOCK=BLOCK)
    z_ref = torch.arange(start, BLOCK+start, dtype=torch.int32, device=device)
    triton.testing.assert_almost_equal(z_tri, z_ref)

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

# ---------------
# test noop
#----------------
def test_noop(device='cuda'):
    @triton.jit
    def kernel(**meta):
        pass
    x = triton.testing.random((1,), dtype=torch.int32, device=device)
    kernel[(1, )](x)