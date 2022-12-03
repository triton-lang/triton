# flake8: noqa: F821,F841
import itertools
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
# TODO: handle bfloat16
dtypes_with_bfloat16 = dtypes  # + ['bfloat16']


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


def check_type_supported(dtype):
    '''
    skip test if dtype is not supported on the current device
    '''
    cc = torch.cuda.get_device_capability()
    if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
        pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")


@pytest.mark.parametrize("dtype_x", [dtype_x for dtype_x in dtypes] + ["bfloat16"])
def test_empty_kernel(dtype_x, device='cuda'):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass
    check_type_supported(dtype_x)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda'):
    check_type_supported(dtype_x)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(z_ref), device=device, dst_type=dtype_x)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE, num_warps=4)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', y_low=None, y_high=None):
    check_type_supported(dtype_x)  # early return if dtype_x is not supported
    check_type_supported(dtype_y)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
    if dtype_z is not None:
        z_ref = z_ref.astype(dtype_z)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
    kernel[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=expr, rtol=0.01)


def _mod_operation_ill_conditioned(dtype_x, dtype_y) -> bool:
    # The result of x % y is ill-conditioned if x % y is much smaller than x.
    # pytorch/CUDA has slightly different (probably better) rounding on
    # remainders than stock LLVM. We currently don't expect to match it
    # bit-for-bit.
    return (dtype_x, dtype_y) in [
        ('int32', 'bfloat16'),
        ('int32', 'float16'),
        ('int32', 'float32'),
        ('int64', 'bfloat16'),
        ('int64', 'float16'),
        ('int64', 'float32'),
        ('int64', 'float64'),
        ('uint16', 'bfloat16'),
        ('uint16', 'float16'),
        ('uint16', 'float32'),
        ('uint32', 'bfloat16'),
        ('uint32', 'float16'),
        ('uint32', 'float32'),
        ('uint64', 'bfloat16'),
        ('uint64', 'float16'),
        ('uint64', 'float32'),
        ('uint64', 'float64'),
    ]

# ---------------
# test binary ops
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['+', '-', '*', '/']  # , '%'] #TODO: handle remainder
    for dtype_x in dtypes_with_bfloat16
    for dtype_y in dtypes_with_bfloat16
])
def test_bin_op(dtype_x, dtype_y, op, device='cuda'):
    expr = f' x {op} y'
    if op == '%' and dtype_x in int_dtypes + uint_dtypes and dtype_y in int_dtypes + uint_dtypes:
        # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
        numpy_expr = 'np.fmod(x, y)'
    elif op in ('/', '%') and dtype_x in ('int16', 'float16', 'bfloat16') and dtype_y in ('int16', 'float16', 'bfloat16'):
        # Triton promotes 16-bit floating-point / and % to 32-bit because there
        # are no native div or FRem operations on float16. Since we have to
        # convert anyway, we may as well take the accuracy bump.
        numpy_expr = f'x.astype(np.float32) {op} y.astype(np.float32)'
    elif (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if op == '%' and _mod_operation_ill_conditioned(dtype_x, dtype_y):
        with pytest.raises(AssertionError, match='Not equal to tolerance'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device)
    elif (op in ('%', '/') and
          ((dtype_x in int_dtypes and dtype_y in uint_dtypes) or
           (dtype_x in uint_dtypes and dtype_y in int_dtypes))):
        with pytest.raises(triton.CompilationError) as exc_info:
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device)
        assert re.match('Cannot use .* because they have different signedness', str(exc_info.value.__cause__))
    else:
        _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device)


@pytest.mark.parametrize("dtype_x, dtype_y",
                         [(dtype_x, dtype_y) for dtype_x in int_dtypes for dtype_y in int_dtypes] +
                         [(dtype_x, dtype_y) for dtype_x in uint_dtypes for dtype_y in uint_dtypes]
                         )
def test_floordiv(dtype_x, dtype_y, device='cuda'):
    # Triton has IEEE, not numpy/torch, semantics for %, and those carry
    # through to //, so we have to use a nonstandard expression to get a
    # reference result for //.
    expr = 'x // y'
    numpy_expr = '((x - np.fmod(x, y)) / y)'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device)


# ---------------
# test bitwise ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['&', '|', '^']
    for dtype_x in dtypes + dtypes_with_bfloat16
    for dtype_y in dtypes + dtypes_with_bfloat16
])
def test_bitwise_op(dtype_x, dtype_y, op, device='cuda'):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if 'float' in dtype_x + dtype_y:
        with pytest.raises(triton.CompilationError) as exc_info:
            _test_binary(dtype_x, dtype_y, expr, numpy_expr='np.array([])', device=device)
        # The CompilationError must have been caused by a C++ exception with this text.
        assert re.match('invalid operands of type', str(exc_info.value.__cause__))
    else:
        _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device)


@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['<<', '>>']
    for dtype_x in int_dtypes + uint_dtypes
    for dtype_y in int_dtypes + uint_dtypes
])
def test_shift_op(dtype_x, dtype_y, op, device='cuda'):
    expr = f'x {op} y'
    bw = max(_bitwidth(dtype_x), _bitwidth(dtype_y))
    dtype_z = f'uint{bw}'
    numpy_expr = f'x.astype(np.{dtype_z}) {op} y.astype(np.{dtype_z})'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, y_low=0, y_high=65)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']


@pytest.mark.parametrize("dtype_x, dtype_y, op, mode_x, mode_y",
                         # real
                         [
                             (dtype_x, dtype_y, op, 'real', 'real')
                             for op in ops
                             for dtype_x in dtypes
                             for dtype_y in dtypes
                         ] +
                         # NaNs
                         [('float32', 'float32', op, mode_x, mode_y)
                             for op in ops
                             for mode_x, mode_y in [('nan', 'real'),
                                                    ('real', 'nan'),
                                                    ('nan', 'nan')]

                          ])
def test_compare_op(dtype_x, dtype_y, op, mode_x, mode_y, device='cuda'):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, mode_x=mode_x, mode_y=mode_y, device=device)


# ---------------
# test where
# ---------------
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16 + ["*int32"])
def test_where(dtype):
    select_ptrs = False
    if dtype == "*int32":
        dtype = "int64"
        select_ptrs = True
    check_type_supported(dtype)

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr,
                     TEST_POINTERS: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        decide = tl.load(cond_ptr + offsets, mask=mask)
        if TEST_POINTERS:
            a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
            b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
        else:
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
        output = tl.where(decide, a, b)
        tl.store(output_ptr + offsets, output, mask=mask)

    SIZE = 1_000
    rs = RandomState(17)
    cond = numpy_random(SIZE, 'bool', rs)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    z = np.where(cond, x, y)

    cond_tri = to_triton(cond, device='cuda')
    x_tri = to_triton(x, device='cuda', dst_type=dtype)
    y_tri = to_triton(y, device='cuda', dst_type=dtype)
    z_tri = to_triton(np.empty(SIZE, dtype=z.dtype), device='cuda', dst_type=dtype)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)
    where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs)
    assert (z == to_numpy(z_tri)).all()


def test_where_broadcast():
    @triton.jit
    def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

        mask = tl.load(cond_ptr + yoffsets)
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    @triton.jit
    def where_scalar_condition(a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = 0
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    SIZE = 32
    dtype = 'float32'
    rs = RandomState(17)
    x = numpy_random((SIZE, SIZE), dtype_str=dtype, rs=rs)
    mask = numpy_random(SIZE, 'bool', rs=rs)
    z = np.where(mask, x, 0)
    cond_tri = to_triton(mask, device="cuda")
    x_tri = to_triton(x, device='cuda', dst_type=dtype)
    z_tri = to_triton(np.empty((SIZE, SIZE), dtype=z.dtype), device='cuda', dst_type=dtype)
    where_kernel[(1,)](cond_tri, x_tri, z_tri, SIZE)
    assert (z == to_numpy(z_tri)).all()
    where_scalar_condition[(1,)](x_tri, z_tri, SIZE)
    z = np.where(0, x, 0)
    assert (z == to_numpy(z_tri)).all()

# # ---------------
# # test unary ops
# # ---------------


@pytest.mark.parametrize("dtype_x, expr", [
    (dtype_x, ' -x') for dtype_x in dtypes_with_bfloat16
] + [
    (dtype_x, ' ~x') for dtype_x in int_dtypes
])
def test_unary_op(dtype_x, expr, device='cuda'):
    _test_unary(dtype_x, expr, device=device)

# # ----------------
# # test math ops
# # ----------------


@pytest.mark.parametrize("expr", [
    'exp', 'log', 'cos', 'sin'
])
def test_math_op(expr, device='cuda'):
    _test_unary('float32', f'tl.{expr}(x)', f'np.{expr}(x) ', device=device)


# # ----------------
# # test indexing
# # ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = triton_gpu.convert_layout %3 : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [
    (f'x[{s}]', d)
    for s in ['None, :', ':, None',
              # TODO: 3D
              #  'None, :, :',
              #  ':, :, None'
              ]
    for d in ['int32', 'uint32', 'uint16']
])
def test_index1d(expr, dtype_str, device='cuda'):
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y + 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


# # ---------------
# # test tuples
# # ---------------


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
@pytest.mark.parametrize("op, dtype_x_str, mode", itertools.chain.from_iterable([
    [
        ('add', 'float16', mode),
        ('add', 'uint32', mode), ('add', 'int32', mode), ('add', 'float32', mode),
        ('max', 'uint32', mode), ('max', 'int32', mode), ('max', 'float32', mode),
        ('min', 'uint32', mode), ('min', 'int32', mode), ('min', 'float32', mode),
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x_str, mode, device='cuda'):
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x)'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = numpy_random((n_programs, ), dtype_str=dtype_x_str, rs=rs)
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, device=device)

    z_tri = to_triton(np.array([neutral], dtype=getattr(np, dtype_x_str)), device=device)
    kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(getattr(np, dtype_x_str))
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == to_numpy(z_tri).item()
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


@pytest.mark.parametrize("shape, axis",
                         [(shape, axis) for shape in [(2, 2), (2, 8), (8, 2), (8, 8), (32, 32)] for axis in [0, 1]])
def test_tensor_atomic_rmw(shape, axis, device="cuda"):
    shape0, shape1 = shape
    # triton kernel

    @triton.jit
    def kernel(Z, X, AXIS: tl.constexpr, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        x = tl.load(X + off0[:, None] * SHAPE1 + off1[None, :])
        z = tl.sum(x, axis=AXIS)
        if AXIS == 1:
            tl.atomic_add(Z + off0, z)
        else:
            tl.atomic_add(Z + off1, z)
    rs = RandomState(17)
    x = numpy_random((shape0, shape1), dtype_str="float32", rs=rs)
    # reference result
    z_ref = np.sum(x, axis=axis, keepdims=False)
    # triton result
    x_tri = to_triton(x, device=device)
    z_shape = (shape0, ) if axis == 1 else (shape1, )
    z_tri = to_triton(np.zeros(z_shape, dtype="float32"), device=device)
    kernel[(1,)](z_tri, x_tri, axis, shape0, shape1)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=1e-4)

def test_atomic_cas():
    # 1. make sure that atomic_cas changes the original value (Lock)
    @triton.jit
    def change_value(Lock):
        tl.atomic_cas(Lock, 0, 1)

    Lock = torch.zeros((1,), device='cuda', dtype=torch.int32)
    change_value[(1,)](Lock)

    assert (Lock[0] == 1)

    # 2. only one block enters the critical section
    @triton.jit
    def serialized_add(data, Lock):
        ptrs = data + tl.arange(0, 128)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass

        tl.store(ptrs, tl.load(ptrs) + 1.0)

        # release lock
        tl.atomic_xchg(Lock, 0)

    Lock = torch.zeros((1,), device='cuda', dtype=torch.int32)
    data = torch.zeros((128,), device='cuda', dtype=torch.float32)
    ref = torch.full((128,), 64.0)
    serialized_add[(64,)](data, Lock)
    triton.testing.assert_almost_equal(data, ref)


# # ---------------
# # test cast
# # ---------------


@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False)
    for dtype_x in dtypes
    for dtype_z in dtypes
] + [
    # TODO:
    # ('float32', 'bfloat16', False),
    # ('bfloat16', 'float32', False),
    ('float32', 'int32', True),
    # TODO:
    ('float32', 'int1', False),
] + [
    (f'uint{x}', f'int{x}', True) for x in [8, 16, 32, 64]
] + [
    (f'int{x}', f'uint{x}', True) for x in [8, 16, 32, 64]
])
def test_cast(dtype_x, dtype_z, bitcast, device='cuda'):
    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    x0 = 43 if dtype_x in int_dtypes else 43.5
    if dtype_x in float_dtypes and dtype_z == 'int1':
        x0 = 0.5
    if dtype_x.startswith('bfloat'):
        x_tri = torch.tensor([x0], dtype=getattr(torch, dtype_x), device=device)
    else:
        x = np.array([x0], dtype=getattr(np, dtype_x))
        x_tri = to_triton(x)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr):
        x = tl.load(X)
        z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        tl.store(Z, z)

    dtype_z_np = dtype_z if dtype_z != 'int1' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((1,), dtype=getattr(torch, dtype_z), device=device)
    else:
        z_tri = to_triton(np.empty((1, ), dtype=getattr(np, dtype_z_np)), device=device)
    kernel[(1, )](x_tri, z_tri, BITCAST=bitcast)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        assert z_tri == z_ref
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        assert to_numpy(z_tri) == z_ref


def test_store_bool():
    """Tests that boolean True is stored as 1"""
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    src = torch.tensor([True, False], dtype=torch.bool, device='cuda')
    n_elements = src.numel()
    dst = torch.empty_like(src)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)

    assert (to_numpy(src).view('uint8') == to_numpy(dst).view('uint8')).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_f8_xf16_roundtrip(dtype):
    """Tests that converting an f8 to f16 and back to f8 doesn't change its value"""
    check_type_supported(dtype)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device='cuda')
    f8 = triton.reinterpret(f8_tensor, tl.float8)
    n_elements = f8_tensor.numel()
    xf16 = torch.empty_like(f8_tensor, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f8, xf16, n_elements, BLOCK_SIZE=1024)

    f8_output_tensor = torch.empty_like(xf16, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    copy_kernel[grid](xf16, f8_output, n_elements, BLOCK_SIZE=1024)

    assert torch.all(f8_tensor == f8_output_tensor)


def test_f16_to_f8_rounding():
    """Takes all float16s, converts them to float8 and back to float16. Checks that the absolute
    error is the minimum over all float8.
    Or the same explanation a bit mathier:
    for all f16 |f16 - fromf8(tof8(f16))| == min over all f8 |f16 - fromf8(f8)|"""
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    # torch.view with a dtype isn't supported in triton's torch yet so use numpy's view
    f16_input_np = (
        np.array(
            range(-int(2 ** (16 - 1)), int(2 ** (16 - 1))), dtype=np.int16,
        )
        .view(np.float16)
    )
    f16_input = torch.tensor(f16_input_np, dtype=torch.float16, device='cuda')
    n_elements = f16_input.numel()
    f8_output_tensor = torch.empty_like(f16_input, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f16_input, f8_output, n_elements, BLOCK_SIZE=1024)

    f16_output = torch.empty_like(f16_input, dtype=torch.float16)
    copy_kernel[grid](f8_output, f16_output, n_elements, BLOCK_SIZE=1024)

    abs_error = torch.abs(f16_input - f16_output)

    all_f8_vals_tensor = torch.tensor(range(2 ** 8), dtype=torch.uint8, device='cuda')
    all_f8_vals = triton.reinterpret(all_f8_vals_tensor, tl.float8)
    all_f8_vals_in_f16 = torch.empty_like(all_f8_vals_tensor, dtype=torch.float16)
    copy_kernel[grid](all_f8_vals, all_f8_vals_in_f16, n_elements=256, BLOCK_SIZE=1024)

    all_finite_f8_vals_in_f16 = all_f8_vals_in_f16[
        torch.isfinite(all_f8_vals_in_f16)
    ]

    min_error = torch.min(
        torch.abs(
            f16_input.reshape((-1, 1))
            - all_finite_f8_vals_in_f16.reshape((1, -1))
        ),
        dim=1,
    )[0]
    # 1.9375 is float8 max
    mismatch = torch.logical_and(
        abs_error != min_error, torch.logical_and(torch.isfinite(f16_input), torch.abs(f16_input) < 1.9375)
    )
    assert torch.all(
        torch.logical_not(mismatch)
    ), f"f16_input[mismatch]={f16_input[mismatch]} f16_output[mismatch]={f16_output[mismatch]} abs_error[mismatch]={abs_error[mismatch]} min_error[mismatch]={min_error[mismatch]}"


# # ---------------
# # test reduce
# # ---------------


def get_reduced_dtype(dtype_str, op):
    if op == 'argmin' or op == 'argmax':
        return 'int32'
    if dtype_str in ['int8', 'uint8', 'int16', 'uint16']:
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


# TODO: [Qingyi] Fix argmin / argmax
@pytest.mark.parametrize("op, dtype_str, shape",
                         [(op, dtype, shape)
                          for op in ['min', 'max', 'sum']
                          for dtype in dtypes_with_bfloat16
                          for shape in [32, 64, 128, 512]])
def test_reduce1d(op, dtype_str, shape, device='cuda'):
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.store(Z, GENERATE_TEST_HERE)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=0)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((shape,), dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'argmin': np.argmin, 'argmax': np.argmax}[op]
    # numpy result
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((1,), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK=shape)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op == 'argmin' or op == 'argmax':
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            np.testing.assert_equal(x[z_ref], x[z_tri])
        else:
            np.testing.assert_equal(z_ref, z_tri)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [
    (op, dtype, (1, 1024), axis) for dtype in dtypes_with_bfloat16
    for op in ['min', 'max', 'sum']
    for axis in [1]
]


# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
#, (32, 64), (64, 128)]
if 'V100' in torch.cuda.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]


reduce_configs2 = [
    (op, 'float32', shape, axis)
    for op in ['min', 'max', 'sum']
    for shape in reduce2d_shapes
    for axis in [0, 1]
]


@pytest.mark.parametrize("op, dtype_str, shape, axis", reduce_configs1 + reduce_configs2)
def test_reduce2d(op, dtype_str, shape, axis, device='cuda'):
    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        z = GENERATE_TEST_HERE
        if AXIS == 1:
            tl.store(Z + range_m, z)
        else:
            tl.store(Z + range_n, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=AXIS)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'argmin': np.argmin, 'argmax': np.argmax}[op]
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    # numpy result
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_tri_dtype_str = 'bfloat16'
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
    else:
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((shape[1 - axis],), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op == 'argmin' or op == 'argmax':
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = np.expand_dims(z_ref, axis=axis)
            z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)

# # ---------------
# # test permute
# # ---------------


@pytest.mark.parametrize("dtype_str, shape, perm",
                         [(dtype, shape, perm)
                          # TODO: bfloat16
                          for dtype in ['float16', 'float32']
                             for shape in [(64, 64), (128, 128)]
                             for perm in [(1, 0)]])
def test_permute(dtype_str, shape, perm, device='cuda'):
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))
    # input
    x = numpy_random(shape, dtype_str=dtype_str)
    # triton result
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    z_tri_contiguous = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         z_tri, z_tri.stride(1), z_tri.stride(0),
                         BLOCK_M=shape[0], BLOCK_N=shape[1])
    pgm_contiguous = kernel[(1, 1)](x_tri, x_tri.stride(1), x_tri.stride(0),
                                    z_tri_contiguous, z_tri_contiguous.stride(0), z_tri_contiguous.stride(1),
                                    BLOCK_M=shape[0], BLOCK_N=shape[1])
    # numpy result
    z_ref = x.transpose(*perm)
    # compare
    triton.testing.assert_almost_equal(z_tri, z_ref)
    triton.testing.assert_almost_equal(z_tri_contiguous, z_ref)
    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx
    ptx = pgm_contiguous.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx

# # ---------------
# # test dot
# # ---------------


@pytest.mark.parametrize("epilogue, allow_tf32, dtype",
                         [(epilogue, allow_tf32, dtype)
                          for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax', 'chain-dot']
                          for allow_tf32 in [True, False]
                          for dtype in ['float16']
                          if not (allow_tf32 and (dtype in ['float16']))])
def test_dot(epilogue, allow_tf32, dtype, device='cuda'):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 80:
        if dtype == 'int8':
            pytest.skip("Only test int8 on devices with sm >= 80")
        elif dtype == 'float32' and allow_tf32:
            pytest.skip("Only test tf32 on devices with sm >= 80")

    M, N, K = 64, 64, 64
    num_warps = 4
    trans_a, trans_b = False, False

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        x = tl.trans(x) if TRANS_A else x
        y = tl.trans(y) if TRANS_B else y
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            # tl.store(Zs, z)
            # tl.debug_barrier()
            z = tl.dot(tl.trans(z.to(tl.float16)), tl.load(Ws))
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    x = numpy_random((K, M) if trans_a else (M, K), dtype_str=dtype, rs=rs) * .1
    y = numpy_random((N, K) if trans_b else (K, N), dtype_str=dtype, rs=rs) * .1
    w = numpy_random((N, N), dtype_str=dtype, rs=rs) * .1
    if allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    # triton result
    z = 1 + numpy_random((M, N), dtype_str=dtype, rs=rs) * .1
    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         TRANS_A=trans_a, TRANS_B=trans_b,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps)
    # torch result
    x_ref = x.T if trans_a else x
    y_ref = y.T if trans_b else y
    z_ref = np.matmul(x_ref, y_ref)
    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        z_ref = np.matmul(z_ref.T, w)
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    # make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx
    if allow_tf32:
        assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
    elif dtype == 'float32':
        assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' not in ptx
    elif dtype == 'int8':
        assert 'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx


# def test_dot_without_load():
#     @triton.jit
#     def kernel(out):
#         pid = tl.program_id(axis=0)
#         a = tl.zeros((32, 32), tl.float32)
#         b = tl.zeros((32, 32), tl.float32)
#         c = tl.zeros((32, 32), tl.float32)
#         c = tl.dot(a, b)
#         pout = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
#         tl.store(pout, c)

#     out = torch.ones((32, 32), dtype=torch.float32, device="cuda")
#     kernel[(1,)](out)

# # ---------------
# # test arange
# # ---------------


@pytest.mark.parametrize("start", [0, 1, 7, 16])
def test_arange(start, device='cuda'):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr,
                START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)
    _kernel[(1,)](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    triton.testing.assert_almost_equal(z_tri, z_ref)

# # ---------------
# # test load
# # ---------------
# # 'bfloat16': torch.bfloat16,
# # Testing masked loads with an intermate copy to shared memory run.


# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
# def test_masked_load_shared_memory(dtype, device='cuda'):
#     check_type_supported(dtype)  # bfloat16 on cc < 80 will not be tested

#     M = 32
#     N = 32
#     K = 16

#     in1 = torch.rand((M, K), dtype=dtype, device=device)
#     in2 = torch.rand((K, N), dtype=dtype, device=device)
#     out = torch.zeros((M, N), dtype=dtype, device=device)

#     @triton.jit
#     def _kernel(in1_ptr, in2_ptr, output_ptr,
#                 in_stride, in2_stride, out_stride,
#                 in_numel, in2_numel, out_numel,
#                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

#         M_offsets = tl.arange(0, M)
#         N_offsets = tl.arange(0, N)
#         K_offsets = tl.arange(0, K)

#         in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
#         in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

#         # Load inputs.
#         x = tl.load(in1_ptr + in_offsets, mask=in_offsets < in_numel)
#         w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < in2_numel)

#         # Without a dot product the memory doesn't get promoted to shared.
#         o = tl.dot(x, w)

#         # Store output
#         output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
#         tl.store(output_ptr + output_offsets, o, mask=output_offsets < in2_numel)

#     pgm = _kernel[(1,)](in1, in2, out,
#                         in1.stride()[0],
#                         in2.stride()[0],
#                         out.stride()[0],
#                         in1.numel(),
#                         in2.numel(),
#                         out.numel(),
#                         M=M, N=N, K=K)

#     reference_out = torch.matmul(in1, in2)
#     triton.testing.allclose(out, reference_out)


@pytest.mark.parametrize("cache", ["", ".ca", ".cg"])
def test_load_cache_modifier(cache):
    src = torch.empty(128, device='cuda')
    dst = torch.empty(128, device='cuda')

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets, cache_modifier=CACHE)
        tl.store(dst + offsets, x)

    pgm = _kernel[(1,)](dst, src, CACHE=cache)
    ptx = pgm.asm['ptx']
    if cache == '':
        assert 'ld.global.ca' not in ptx
        assert 'ld.global.cg' not in ptx
    if cache == '.cg':
        assert 'ld.global.cg' in ptx
        assert 'ld.global.ca' not in ptx
    if cache == '.ca':
        assert 'ld.global.ca' in ptx
        assert 'ld.global.cg' not in ptx


@pytest.mark.parametrize("N", [16, 10, 11, 1024])
def test_vectorization(N):
    src = torch.empty(1024, device='cuda')
    dst = torch.empty(1024, device='cuda')

    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)
    pgm = _kernel[(1,)](dst, src, N=N, BLOCK_SIZE=src.shape[0])
    ptx = pgm.asm["ptx"]
    if N % 16 == 0:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.b32" in ptx
    # triton.testing.assert_almost_equal(dst, src[:N])
# # ---------------
# # test store
# # ---------------

# # ---------------
# # test if
# # ---------------

# # ---------------
# # test for
# # ---------------

# # ---------------
# # test while
# # ---------------

# # ---------------
# # test default
# # ---------------
# # TODO: can't be local to test_default


@triton.jit
def _impl(value=10):
    return value


def test_default():
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device='cuda')
    ret1 = torch.zeros(1, dtype=torch.int32, device='cuda')

    @triton.jit
    def _kernel(ret0, ret1, value):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1,)](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

# # ---------------
# # test noop
# # ----------------


def test_noop(device='cuda'):
    @triton.jit
    def kernel(x):
        pass
    x = to_triton(numpy_random((1,), dtype_str='int32'), device=device)
    kernel[(1, )](x)


@pytest.mark.parametrize("value, value_type", [
    (-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
    (2**31, 'u32'), (2**32 - 1, 'u32'), (2**32, 'i64'), (2**63 - 1, 'i64'),
    (-2**63, 'i64'), (2**63, 'u64'), (2**64 - 1, 'u64')
])
def test_value_specialization(value: int, value_type: str, device='cuda') -> None:
    spec_type = None

    def cache_hook(*args, **kwargs):
        nonlocal spec_type
        spec_type = kwargs["compile"]["signature"][0]
    JITFunction.cache_hook = cache_hook

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device='cuda')
    pgm = kernel[(1, )](value, x)

    JITFunction.cache_hook = None
    assert spec_type == value_type

# # --------------------
# # value specialization
# # --------------------


@pytest.mark.parametrize(
    "value, overflow",
    [(2**64 - 1, False), (2**64, True), (-2**63, False), (-2**63 - 1, True)]
)
def test_value_specialization_overflow(value: int, overflow: bool, device='cuda') -> None:

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device='cuda')

    if overflow:
        with pytest.raises(OverflowError):
            kernel[(1, )](value, x)
    else:
        kernel[(1, )](value, x)


# # ----------------
# # test constexpr
# # ----------------

@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '<', '>'])
@pytest.mark.parametrize("is_lhs_constexpr", [False, True])
@pytest.mark.parametrize("is_rhs_constexpr", [True, False])
def test_bin_op_constexpr(op, is_lhs_constexpr, is_rhs_constexpr):

    @triton.jit
    def kernel(Z, X, Y):
        x = tl.load(X)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z, z)

    x_str = "3.14" if is_lhs_constexpr else "x"
    y_str = "4.13" if is_rhs_constexpr else "y"
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{x_str} {op} {y_str}"})
    x = numpy_random((1,), dtype_str="float32")
    y = numpy_random((1,), dtype_str="float32")
    z = np.array(eval(f"{x_str} {op} {y_str}"))
    x_tri = to_triton(x)
    y_tri = to_triton(y)
    z_tri = to_triton(np.empty((1,), dtype=z.dtype))
    kernel[(1,)](z_tri, x_tri, y_tri)
    np.testing.assert_allclose(z, to_numpy(z_tri))


def test_constexpr_shape():

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32))
    kernel[(1,)](x_tri)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256))


def test_constexpr_scalar_shape():

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32))
    kernel[(1,)](x_tri, 32)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256) % 8)

# # -------------
# # test call
# # -------------


@triton.jit
def val_multiplier(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        vec = val_multiplier(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


def test_call():

    @triton.jit
    def kernel(ptr, n_elements, num1, num2):
        vecmul_kernel(ptr, n_elements, num1)
        vecmul_kernel(ptr, n_elements, num2)

    size = 1024
    rand_val = numpy_random((size,), dtype_str="float32")
    rand_val_tri = to_triton(rand_val, device='cuda')
    kernel[(size // 128,)](rand_val_tri, size, 3, 5)

    ans = rand_val * 1 * 2 * 1 * 2 * 3 * 4
    np.testing.assert_equal(to_numpy(rand_val_tri), ans)

# # -------------
# # test if
# # -------------


def test_if():

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if pid % 2:
            tl.store(Ret, tl.load(XTrue))
        else:
            tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device='cuda')
    x_true = torch.tensor([3.14], dtype=torch.float32, device='cuda')
    x_false = torch.tensor([1.51], dtype=torch.float32, device='cuda')
    ret = torch.empty(1, dtype=torch.float32, device='cuda')
    kernel[(1,)](cond, x_true, x_false, ret)


def test_num_warps_pow2():
    dst = torch.empty(128, device='cuda')

    @triton.jit
    def _kernel(dst):
        pass

    with pytest.raises(AssertionError, match='must be a power of 2'):
        _kernel[(1,)](dst=dst, num_warps=3)
    _kernel[(1,)](dst=dst, num_warps=1)
    _kernel[(1,)](dst=dst, num_warps=2)
    _kernel[(1,)](dst=dst, num_warps=4)

# # -------------
# # test extern
# # -------------


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('int32', 'libdevice.ffs', ''),
                          ('float32', 'libdevice.pow', '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'),
                          ('float64', 'libdevice.norm4d', '')])
def test_libdevice_tensor(dtype_str, expr, lib_path):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)

    if expr == 'libdevice.ffs':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.ffs(x)'})
        y_ref = np.zeros(shape, dtype=x.dtype)
        for i in range(shape[0]):
            y_ref[i] = (int(x[i]) & int(-x[i])).bit_length()
    elif expr == 'libdevice.pow':
        # numpy does not allow negative factors in power, so we use abs()
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.pow(x, x)'})
        y_ref = np.power(x, x)
    elif expr == 'libdevice.norm4d':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.norm4d(x, x, x, x)'})
        y_ref = np.sqrt(4 * np.power(x, 2))

    x_tri = to_triton(x)
    # triton result
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    if expr == 'libdevice.ffs':
        np.testing.assert_equal(y_ref, to_numpy(y_tri))
    else:
        np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('float32', 'libdevice.pow', '')])
def test_libdevice_scalar(dtype_str, expr, lib_path):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = X
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((1,), dtype_str=dtype_str, rs=rs)
    y_ref = np.zeros(shape, dtype=x.dtype)

    # numpy does not allow negative factors in power, so we use abs()
    x = np.abs(x)
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.libdevice.pow(x, x)'})
    y_ref[:] = np.power(x, x)

    # triton result
    x_tri = to_triton(x)[0].item()
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)
