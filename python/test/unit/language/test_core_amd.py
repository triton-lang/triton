# flake8: noqa: F821,F841
import triton.language.semantic
import itertools
import os
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.common.build import is_hip
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']


def hip_skip():
    import inspect
    return pytest.skip(f"Skipping {inspect.stack()[1][3]}!")

if is_hip():
    GPU_DIALECT = "triton_gpu"
    THREADS_PER_WARP = 64
else:
    GPU_DIALECT = "triton_gpu"
    THREADS_PER_WARP = 32


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
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
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
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
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


def check_type_supported(dtype, device='cuda'):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")

@pytest.mark.parametrize("dtype_x", list(dtypes) + ["bfloat16"])
def test_empty_kernel(dtype_x, device='cuda'):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass
    check_type_supported(dtype_x)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


def test_program_functions(dtype_x = 'float16', device='cuda'):
    SIZE = 128
    # os.environ["TRITON_DEBUG"] = "1"
    @triton.jit
    def kernel(X, Y, SIZE: tl.constexpr):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_z = tl.program_id(2)
        num_pids_x = tl.num_programs(0)
        num_pids_y = tl.num_programs(1)
        num_pids_z = tl.num_programs(2)

        pid = pid_z * (num_pids_x * num_pids_y) + pid_y * num_pids_x + pid_x
        val = (pid_x == 0)
        val = val and (pid_y == 0)
        val = val and (pid_z == 0 or pid_z == 1)
        val = val and (num_pids_x == 1)
        val = val and (num_pids_y == 1)
        val = val and (num_pids_z == 2)
        tl.store(Y + pid, val)

    check_type_supported(dtype_x)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    gold = torch.ones((2,), device = device, dtype = torch.bool)
    y = torch.zeros((2,), device = device, dtype = torch.bool)
    kernel[(1, 1, 2)](x, y, SIZE=SIZE, num_warps=4)
    np.testing.assert_allclose(to_numpy(gold), to_numpy(y))

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
    for op in ['+', '-', '*', '/', '%']
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


def test_unsigned_name_mangling(device='cuda'):
    # Test that uint32 and int32 are mangled differently by the compiler
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(
        to_triton(np.empty_like(e), device=device)
        for e in expect
    )
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


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
    if dtype_x.startswith('int'):
        dtype_z = f'int{bw}'
    else:
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
                     TEST_POINTERS: tl.constexpr,
                     TEST_SCALAR_POINTERS: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        decide = tl.load(cond_ptr + offsets, mask=mask)
        if TEST_SCALAR_POINTERS:
            ptr = tl.where(tl.load(cond_ptr), a_ptr, b_ptr)
            output = tl.load(ptr + offsets, mask=mask)
        else:
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
    where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs, TEST_SCALAR_POINTERS=False)
    assert (z == to_numpy(z_tri)).all()
    if select_ptrs:
        where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs, TEST_SCALAR_POINTERS=True)
        z = np.where(cond[0], x, y)
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

# ---------------
# test unary ops
# ---------------


@pytest.mark.parametrize("dtype_x, expr", [
    (dtype_x, ' -x') for dtype_x in dtypes_with_bfloat16
] + [
    (dtype_x, ' ~x') for dtype_x in int_dtypes
])
def test_unary_op(dtype_x, expr, device='cuda'):
    _test_unary(dtype_x, expr, device=device)

# ----------------
# test math ops
# ----------------


@pytest.mark.parametrize("expr", [
    'exp', 'log', 'cos', 'sin'
])
def test_math_op(expr, device='cuda'):
    _test_unary('float32', f'tl.{expr}(x)', f'np.{expr}(x) ', device=device)

# ----------------
# test abs
# ----------------


@pytest.mark.parametrize("dtype_x", [
    (dtype_x)
    for dtype_x in dtypes_with_bfloat16
])
def test_abs(dtype_x, device='cuda'):
    _test_unary(dtype_x, 'tl.abs(x)', 'np.abs(x) ', device=device)

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


# TODO: handle `%4 = triton_gpu.convert_layout %3 : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [
    (f'x[{s}]', d)
    for s in ['None, :', ':, None',
              'None, :, :',
              ':, :, None']
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
@pytest.mark.parametrize("op, dtype_x_str, mode", itertools.chain.from_iterable([
    [
        ('add', 'float16', mode),
        ('add', 'uint32', mode), ('add', 'int32', mode), ('add', 'float32', mode),
        ('max', 'uint32', mode), ('max', 'int32', mode), ('max', 'float32', mode),
        ('min', 'uint32', mode), ('min', 'int32', mode), ('min', 'float32', mode),
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']]))
def test_atomic_rmw(op, dtype_x_str, mode, device='cuda'):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        if dtype_x_str == 'float16':
            pytest.skip("Only test atomic float16 ops on devices with sm >= 70")
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
    x = np.array([2**i for i in range(n_programs)], dtype=getattr(np, dtype_x_str))
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


def test_tensor_atomic_rmw_block(device="cuda"):
    shape = (8, 8)

    @triton.jit
    def kernel(X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        offs = off0[:, None] * SHAPE1 + off1[None, :]
        val = offs.to(tl.float32)
        x = X + offs
        tl.atomic_min(x, val)
    x = torch.ones((8, 8), device=device, dtype=torch.float32)
    kernel[(2,)](x, shape[0], shape[1])
    assert torch.min(x).item() == 0.0


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
    np.testing.assert_allclose(to_numpy(data), to_numpy(ref))


# ---------------
# test cast
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False)
    for dtype_x in dtypes
    for dtype_z in dtypes
] + [
    ('float32', 'bfloat16', False),
    ('bfloat16', 'float32', False),
    ('float32', 'int32', True),
    ('float32', 'int1', False),
    ('int8', 'bfloat16', False),
] + [
    (f'uint{x}', f'int{x}', True) for x in [8, 16, 32, 64]
] + [
    (f'int{x}', f'uint{x}', True) for x in [8, 16, 32, 64]
])
def test_cast(dtype_x, dtype_z, bitcast, device):
    # bfloat16 on cc < 80 will not be tested
    check_type_supported(dtype_x, device)
    check_type_supported(dtype_z, device)

    size = 1024
    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    if dtype_x.startswith('bfloat'):
        x_tri = torch.randn(size, dtype=getattr(torch, dtype_x), device=device)
    else:
        x = numpy_random(size, dtype_str=dtype_x, low=-10, high=10) * 10
        # Triton clamps negative values to zero, while numpy wraps around
        # intmax, so avoid negatives for now.
        # TODO: figure out which one should actually be happening, and test it
        if dtype_z in uint_dtypes:
            x = np.absolute(x)
        x_tri = to_triton(x, device=device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr, SIZE: tl.constexpr):
        x_ptr = X + tl.arange(0, SIZE)
        z_ptr = Z + tl.arange(0, SIZE)
        x = tl.load(x_ptr)
        z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        tl.store(z_ptr, z)

    dtype_z_np = dtype_z if dtype_z != 'int1' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((size,), dtype=getattr(torch, dtype_z), device=device)
    else:
        z_tri = to_triton(np.empty((size, ), dtype=getattr(np, dtype_z_np)), device=device)
    kernel[(1, )](x_tri, z_tri, BITCAST=bitcast, SIZE=size, num_warps=1)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        torch.testing.assert_close(z_ref, z_tri, rtol=0, atol=0)
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0, atol=0)


@pytest.mark.parametrize("dtype_str", list(torch_dtypes))
def test_store_constant(dtype_str):
    check_type_supported(dtype_str)

    """Tests that boolean True is stored as 1"""
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        output = GENERATE_TEST_HERE
        tl.store(output_ptr + offsets, output, mask=mask)

    triton_dtype_str = 'uint8' if dtype_str == 'bool' else dtype_str
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.zeros([BLOCK_SIZE], dtype=tl.{triton_dtype_str}) + 1'})
    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, dtype_str), device='cuda')
    output = torch.zeros([block_size], dtype=getattr(torch, dtype_str), device='cuda')
    kernel[(1,)](output, block_size, BLOCK_SIZE=block_size)

    assert torch.all(output == ref)


def test_load_store_same_ptr():
    @triton.jit()
    def kernel(in_out_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(in_out_ptr + pid)
        out = x * 2
        tl.store(in_out_ptr + pid, out)

    for _ in range(1000):
        x = torch.ones((65536,), device="cuda", dtype=torch.float32)
        kernel[(65536,)](x, num_warps=16)
        assert torch.all(x == 2)


def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(exp == 0,
                         # subnormal
                         ((-1.0) ** sign) * (2.0 ** (1 - exp_bias)) * (frac / (2.0 ** dtype.fp_mantissa_width)),
                         # normal
                         ((-1.0) ** sign) * (2.0 ** (exp - exp_bias)) * (1.0 + frac / (2.0 ** dtype.fp_mantissa_width))).float()

    extended_exp = ((1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4nv, tl.float8e4b15]:
        # float8e4m3nv does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    elif dtype in [tl.float8e4b8, tl.float8e5b16]:
        output[fp==0b10000000] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width))).view(torch.float32),
                             output)
    return output


@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_convert_float16_to_float32(in_dtype, device):
    """Tests that check convert_float_to_float32 function"""
    check_type_supported(in_dtype, device)

    f16_input = torch.tensor(range(-int(2 ** (16 - 1)), int(2 ** (16 - 1))), dtype=torch.int16).view(in_dtype)
    f32_output = convert_float_to_float32(f16_input)

    nan = f16_input.isnan()
    assert torch.all(f32_output[nan].isnan())
    inf = f16_input.isinf()
    assert torch.all(f32_output[inf].isinf())
    other = torch.logical_not(torch.logical_or(nan, inf))
    assert torch.all(f16_input[other] == f32_output[other])


def serialize_fp8(np_data, in_dtype):
    if in_dtype == tl.float8e4b15x4:
        # triton's f8e4b15 format is optimized for software emulation
        # as a result, each pack of 4xfp8 values:
        # s0b0s1b1s2b2s3b3 (for s, b sign and bits respectively)
        # is actually internally stored as
        # s0s2b0b2s1s3b1b3
        # we apply the conversion here
        f8x4 = np_data.view(np.uint32)
        s = [(f8x4 & (0x80000000 >> i)) << i for i in range(0, 32, 8)]
        b = [(f8x4 & (0x7f000000 >> i)) << i for i in range(0, 32, 8)]
        signs = (s[0] >> 0) | (s[1] >> 16) | (s[2] >> 1) | (s[3] >> 17)
        bits = (b[0] >> 1) | (b[1] >> 17) | (b[2] >> 8) | (b[3] >> 24)
        # tensor of triton fp8 data
        return (signs | bits).view(np.int8)
    else:
        return np_data

# inverse of `serialize_fp8`


def deserialize_fp8(np_data, in_dtype):
    if in_dtype == tl.float8e4b15x4:
        f8x4 = np_data.view(np.uint32)
        s = [(f8x4 & (0x80000000 >> i)) << i for i in [0, 16, 1, 17]]
        b = [(f8x4 & (0x7f000000 >> i)) << i for i in [1, 17, 8, 24]]
        signs = (s[0] >> 0) | (s[1] >> 8) | (s[2] >> 16) | (s[3] >> 24)
        bits = (b[0] >> 0) | (b[1] >> 8) | (b[2] >> 16) | (b[3] >> 24)
        return (signs | bits).view(np.int8)
    else:
        return np_data


@pytest.mark.parametrize("in_dtype", [tl.float8e4b15, 
                                      tl.float8e4b15x4, 
                                      tl.float8e4b8, 
                                      tl.float8e5, 
                                      tl.float8e5b16])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_fp8_fpN_roundtrip(in_dtype, out_dtype, device):
    """
    For all possible float8 values (ref_fp8 = range(0, 256)), test that:
        - conversion tri_fp16 = convert(input=ref_fp8, out=out_dtype) matches the reference
        - conversion tri_fp8 = convert(input=tri_fp16, out=out_dtype) matches the original
    this is only possible if both conversions are correct
    """
    check_type_supported(in_dtype, device)
    check_type_supported(out_dtype, device)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    # initialize array containing all possible f8 values except NaN
    ref_fp8 = np.array(range(-128, 128), dtype=np.int8)
    is_nan = (ref_fp8 & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    exp_mask = 0b01111111 ^ ((1 << in_dtype.fp_mantissa_width) - 1)
    is_subnormal = np.logical_or((ref_fp8 & exp_mask) == 0, (ref_fp8 & exp_mask) == exp_mask)

    ref_fp8[is_nan] = 0
    ref_fp8[is_subnormal] = 0
    tri_fp8 = torch.from_numpy(serialize_fp8(ref_fp8, in_dtype)).cuda()

    tri_fp16 = torch.empty(256, dtype=out_dtype, device="cuda")
    copy_kernel[(1,)](triton.reinterpret(tri_fp8, in_dtype), tri_fp16, tri_fp16.shape[0], BLOCK_SIZE=1024)

    ref_fp8 = torch.from_numpy(ref_fp8).cuda()
    ref_fp16 = convert_float_to_float32(ref_fp8, in_dtype)
    assert torch.all(tri_fp16[~is_subnormal] == ref_fp16[~is_subnormal])

    ref_fp8 = torch.empty_like(tri_fp16, dtype=torch.int8)
    copy_kernel[(1,)](tri_fp16, triton.reinterpret(ref_fp8, in_dtype), tri_fp16.shape[0], BLOCK_SIZE=1024)
    assert torch.all(tri_fp8 == ref_fp8)


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.float32: torch.float32,
}

if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz


def gen_input(M, N, d_type, seed, device='cuda'):
    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if d_type == tl.float16:
        input = torch.randn((M, N), dtype=torch.float16, device=device)
        input_f16 = input
    else: # d_type is float8
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda') * 10
        if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
            (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) :
            input = raw_data.to(tl_to_torch_types[d_type])
            input_f16 = input.to(torch.float16)
        else:
            f8_tensor = raw_data.to(torch.int8)
            # keep only two bits of exponent to avoid overflow
            f8_tensor = f8_tensor & 0b00111111
            input = triton.reinterpret(f8_tensor, d_type)
            input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            n_elements = raw_data.numel()
            copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)
    return input, input_f16


@pytest.mark.parametrize("M, N, K, a_type, b_type, out_dtype",
                        [(*shape, *ab_type, out_dtype)
                          for shape in [[128, 256, 32],
                                        [128, 16, 32],
                                        [32, 128, 64],
                                        [128, 128, 64],
                                        [64, 128, 128],
                                        [32, 128, 64],
                                        [64, 64, 32],
                                        [32, 32, 128],
                                        [128, 128, 64],
                                        [64, 128, 128]]
                          for ab_type in [[tl.float8e4nv, tl.float16],
                                           [tl.float8e4b15, tl.float16],
                                           [tl.float8e4b15x4, tl.float16],
                                           [tl.float8e5, tl.float16],
                                           [tl.float8e4b8, tl.float16],
                                           [tl.float8e5b16, tl.float16],
                                           [tl.float16, tl.float8e5b16],
                                           [tl.float16, tl.float8e4b8],
                                           [tl.float16, tl.float8e4nv],
                                           [tl.float16, tl.float8e4b15],
                                           [tl.float16, tl.float8e4b15x4],
                                           [tl.float16, tl.float8e5]]
                          for out_dtype in [torch.float16, torch.float32]
                        ])
def test_gemm_fp816_mixed_inputs(M, N, K, a_type, b_type, out_dtype, device = 'cuda'):
    check_type_supported(out_dtype, device)

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        compute_type:tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(compute_type)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def matmul(a, b, c_type):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        M, K = a.shape
        K, N = b.shape

        if c_type == torch.float16:
            comp_type = tl.float16
        else:
            comp_type = tl.float32


        c = torch.empty((M, N), device = a.device, dtype=c_type)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            compute_type = comp_type,
            BLOCK_SIZE_M=32, 
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=4,
            num_stages=1,
            num_warps=2,
        )
        return c

    a, a_f16 = gen_input(M, K, a_type, 11, device=device)
    b, b_f16 = gen_input(K, N, b_type, 22, device=device)

    # call torch function to compute gold
    golden = torch.matmul(a_f16, b_f16)

    c = matmul(a, b, out_dtype)
    torch.testing.assert_close(c.to(golden.dtype), golden, rtol=1e-2, atol=1e-2)


# @pytest.mark.skip(reason="Pytorch does not support the following types, so need to skip for now")
@pytest.mark.parametrize("M, N, K, a_type, b_type, out_dtype",
                        [(*shape, *ab_type, out_dtype)
                          for shape in [[128, 256, 32],
                                        [128, 16, 32],
                                        [32, 128, 64],
                                        [128, 128, 64],
                                        [64, 128, 128],
                                        [32, 128, 64],
                                        [64, 64, 32],
                                        [32, 32, 128],
                                        [128, 128, 64],
                                        [64, 128, 128]]
                          for ab_type in [[tl.float8e4b8, tl.float8e4b8],
                                          [tl.float8e5b16, tl.float8e4b8],
                                          [tl.float8e4b8, tl.float8e5b16],
                                          [tl.float8e5b16, tl.float8e5b16]]
                          for out_dtype in [torch.float32]
                        ])
def test_gemm_amd_fp8_inputs(M, N, K, a_type, b_type, out_dtype, device = 'cuda'):
    check_type_supported(out_dtype, device)

    backend = triton.common.backend.get_backend("hip")
    if backend.get_matrix_core_version() != 3:
        pytest.skip("fp8 data type is not available on hardware")

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        compute_type:tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(compute_type)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def matmul(a, b, c_type):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        M, K = a.shape
        K, N = b.shape

        if c_type == torch.float16:
            comp_type = tl.float16
        else:
            comp_type = tl.float32


        c = torch.empty((M, N), device = a.device, dtype=c_type)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            compute_type = comp_type,
            BLOCK_SIZE_M=32, 
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=4,
            num_stages=1,
            num_warps=2,
        )

        return c

    a, a_f16 = gen_input(M, K, a_type, 21, device=device)
    b, b_f16 = gen_input(K, N, b_type, 22, device=device)

    # call torch function to compute gold
    golden = torch.matmul(a_f16, b_f16)
    c = matmul(a, b, out_dtype)

    torch.testing.assert_close(golden, c.to(golden.dtype), rtol=1e-2, atol=2e-2)


# ---------------
# test reduce
# ---------------
def get_reduced_dtype(dtype_str, op):
    if op in ('argmin', 'argmax'):
        return 'int32'
    if dtype_str in ['int8', 'uint8', 'int16', 'uint16']:
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


@pytest.mark.parametrize("op, dtype_str, shape",
                         [(op, dtype, shape)
                          for op in ['min', 'max', 'sum', 'argmin', 'argmax']
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
    z_dtype_str = 'int32' if op in ('argmin', 'argmax') else dtype_str
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
        if op in ('argmin', 'argmax'):
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
# , (32, 64), (64, 128)]
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
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested

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
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = np.expand_dims(z_ref, axis=axis)
            z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)

# ---------------
# test scan
# ---------------


scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [
    (op, type, shape, axis, num_warps)
    for num_warps in [4, 16]
    for type in ['int32', 'float32']
    for axis in [1, 0]
    for shape in scan2d_shapes
    for op in ['cumsum', 'cumprod']
]


@pytest.mark.parametrize("op, dtype_str, shape, axis, num_warps", scan_configs)
def test_scan2d(op, dtype_str, shape, axis, num_warps, device):
    check_type_supported(dtype_str, device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        z = GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis={axis})'})
    # input
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device)
    numpy_op = {'cumsum': np.cumsum, 'cumprod': np.cumprod}[op]
    z_dtype_str = dtype_str
    z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(z, device=device)
    kernel[(1,)](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)
    z_tri = to_numpy(z_tri)
    # compare
    if dtype_str == 'float32':
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)





# ---------------
# test permute
# ---------------


@pytest.mark.parametrize("dtype_str, shape, perm",
                         [(dtype, shape, perm)
                          # TODO: bfloat16
                          for dtype in ['float8e4b15', 'float16', 'float32']
                             for shape in [(64, 64), (128, 128)]
                             for perm in [(1, 0)]])
def test_permute(dtype_str, shape, perm, device='cuda'):
    check_type_supported(dtype_str)  # bfloat16 on cc < 80 will not be tested
    if torch.version.hip is not None:
        if (dtype_str, shape) == ('float32', (128, 128)):
            pytest.skip("Not supported: memory out of resource.")

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
    if dtype_str == 'float8e4b15':
        ty = tl.float8e4b15
        z_ref = serialize_fp8(deserialize_fp8(x, ty).T.copy(), ty)
        z_tri = z_tri.base
        z_tri_contiguous = z_tri_contiguous.base
    else:
        z_ref = x.transpose(*perm)
    # compare
    np.testing.assert_allclose(to_numpy(z_tri), z_ref)
    np.testing.assert_allclose(to_numpy(z_tri_contiguous), z_ref)
    if torch.version.hip is None:
        # parse ptx to make sure ld/st are vectorized
        ptx = pgm.asm['ptx']
        assert 'ld.global.v4' in ptx
        assert 'st.global.v4' in ptx
        ptx = pgm_contiguous.asm['ptx']
        assert 'ld.global.v4' in ptx
        assert 'st.global.v4' in ptx

# ---------------
# test dot
# ---------------


@pytest.mark.parametrize("M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, in_dtype, out_dtype, non_k_dim",
                         # FMA Test Dot tests
                         [(*shape, 4, False, False, epilogue, allow_tf32, in_dtype, out_dtype, 0)
                          for shape in [(64, 64, 64), (16, 16, 16)]
                          for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax', 'chain-dot']
                          for allow_tf32 in [True, False]
                          for in_dtype, out_dtype in [('float16', 'float16'),
                                                      ('float16', 'float32'),
                                                      ('float32', 'float32')]
                          if not (allow_tf32 and (in_dtype in ['float16']))] +

                         [(*shape_nw, col_a, col_b, 'none', allow_tf32, in_dtype, out_dtype, 0)
                          for shape_nw in [[128, 256, 32, 8],
                                           [128, 16, 32, 4],
                                           [32, 128, 64, 4],
                                           [128, 128, 64, 4],
                                           [64, 128, 128, 4],
                                           [32, 128, 64, 2],
                                           [64, 64, 32, 4],
                                           [32, 32, 128, 16],
                                           [128, 128, 64, 2],
                                           [64, 128, 128, 2]]
                          for allow_tf32 in [True]
                          for col_a in [True, False]
                          for col_b in [True, False]
                          for in_dtype, out_dtype in [('int8', 'int8'),
                                                      ('float16', 'float16'),
                                                      ('float16', 'float32'),
                                                      ('float32', 'float32')]]
                         if triton.common.backend.get_backend("hip").get_matrix_core_version() == 0 else
                         # MFMA Test Dot tests
                         [(*shape, 2, False, False, epilogue, allow_tf32, in_dtype, out_dtype, non_k_dim)
                          for shape in [(64, 64, 64), (32, 32, 32), (16, 16, 16)]
                          for epilogue in ['none', 'trans', 'add-matrix', 'chain-dot', 'softmax']
                          for allow_tf32 in [True, False]
                          for in_dtype, out_dtype in [('float16', 'float16'),
                                                      ('bfloat16', 'float32'),
                                                      ('float8e5m2fnuz', 'float32'),
                                                      ('float8e4m3fnuz', 'float32'),
                                                      ('float16', 'float32'),
                                                      ('float32', 'float32')]
                          for non_k_dim in [0, 16, 32]
                          if not (allow_tf32 and (in_dtype in ['float16']))] +

                         [(*shape_nw, col_a, col_b, 'none', allow_tf32, in_dtype, out_dtype, non_k_dim)
                          for shape_nw in [[128, 128, 32, 2],
                                           [128, 16, 32, 4],
                                           [128, 128, 64, 2],
                                           [128, 32, 32, 2],
                                           [128, 32, 64, 2],
                                           [32, 32, 32, 4],
                                           [32, 32, 64, 4],
                                           [32, 32, 128, 4],
                                           [32, 32, 256, 4],
                                           [64, 64, 32, 4],
                                           [64, 64, 64, 4],
                                           [64, 64, 128, 4],
                                           [32, 128, 64, 2],
                                           [64, 128, 128, 2],
                                           [32, 32, 64, 1],
                                           [32, 128, 64, 2],
                                           [64, 128, 128, 2],
                                           [32, 256, 32, 8],
                                           [32, 256, 32, 2],
                                           [64, 32, 32, 2],
                                           [256, 32, 32, 2],
                                           [256, 32, 32, 4],
                                           ]
                          for allow_tf32 in [False, True]
                          for col_a in [True, False]
                          for col_b in [True, False]
                          for in_dtype in ['int8', 'bfloat16', 'float16', 'float32']
                          for out_dtype in [None]
                          for non_k_dim in [0, 16, 32]])
def test_dot(M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, in_dtype, out_dtype, non_k_dim, device='cuda'):
    capability = torch.cuda.get_device_capability()

    if torch.version.hip is not None:
        # TODO consider reenabling this tests when fp8 casing is fixed
        if M == 16 and N == 16 and K == 16 and "float8" in in_dtype:
            pytest.skip("triton do not generate MFMA instructions for given block size")

        # set capability to large number to jump over check below
        # check are not relevant to amd gpu, left them for smaller diff between test_core.py and test_core_amd.py tests
        if (M, N, K) == (128, 256, 32):
            pytest.skip("Out of resources")
        capability = (100, 100)
        if out_dtype is None:
            if in_dtype in float_dtypes:
                out_dtype = "float32"
            else:
                out_dtype = "int32"
        if non_k_dim == 32 and (M < 32 or N < 32):
            pytest.skip("incompatible non_k_dim == 32 with MN sizes")

    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8:
        if in_dtype == 'int8':
            pytest.skip("Only test int8 on devices with sm >= 80")
        elif in_dtype == 'float32' and allow_tf32:
            pytest.skip("Only test tf32 on devices with sm >= 80")
    if capability[0] == 7:
        if (M, N, K, num_warps) == (128, 256, 32, 8):
            pytest.skip("shared memory out of resource")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               in_dtype: tl.constexpr,
               out_dtype: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               COL_A: tl.constexpr, COL_B: tl.constexpr):
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
        # if in_dtype is tl.float8e4b15 or in_dtype is tl.float8e5:
        # TODO change types when they are available
        if in_dtype is tl.float8e5b16 or in_dtype is tl.float8e4b8:
            x = x.to(in_dtype)
            y = y.to(in_dtype)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32, out_dtype=out_dtype)
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
            num = tl.exp(z.to(tl.float32)).to(max.dtype)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(Ws)
            z = tl.dot(z.to(w.dtype), w, out_dtype=out_dtype)
        tl.store(Zs, z)
    # input
    if in_dtype == "int8":
        effective_in_dtype = tl.int8
    elif in_dtype == "float32":
        effective_in_dtype = tl.float32
    elif in_dtype == "float16":
        effective_in_dtype = tl.float16
    elif in_dtype == "bfloat16":
        effective_in_dtype = tl.bfloat16
    elif in_dtype == "float8e5m2fnuz":
        # TODO change types when they are available
        # effective_in_dtype = tl.float8e5
        effective_in_dtype = tl.float8e5b16
        in_dtype = "float32"
    elif in_dtype == "float8e4m3fnuz":
        # TODO change types when they are available
        # effective_in_dtype = tl.float8e4b15
        effective_in_dtype = tl.float8e4b8
        in_dtype = "float32"
    else:
        assert("unexpected in dtype")

    rs = RandomState(17)
    if col_a:
        x = numpy_random((K, M), dtype_str=in_dtype, rs=rs).T
    else:
        x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    if col_b:
        y = numpy_random((N, K), dtype_str=in_dtype, rs=rs).T
    else:
        y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    w = numpy_random((N, N), dtype_str=in_dtype, rs=rs)
    if 'int' not in in_dtype:
        x *= .1
        y *= .1
    if in_dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    if effective_in_dtype.is_fp8():
        x = x + 1
        y = y + 1
        if effective_in_dtype.is_fp8e5b16():
            mask = 0b111111000110 << 20
        else:
            mask = 0b101111000111 << 20
        x = (x.view('uint32') & np.uint32(mask)).view('float32')
        y = (y.view('uint32') & np.uint32(mask)).view('float32')
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    if in_dtype == 'bfloat16':
        x_tri = x_tri.to(torch.bfloat16)
        y_tri = y_tri.to(torch.bfloat16)
        w_tri = w_tri.to(torch.bfloat16)
    # triton result
    if out_dtype == 'int8':
        z = 1 + numpy_random((M, N), dtype_str='int32', rs=rs)
    else:
        z = 1 + numpy_random((M, N), dtype_str=in_dtype, rs=rs) * .1

    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])

    if out_dtype == 'int8':
        out_dtype = tl.int8
    elif out_dtype == 'float16' and epilogue != 'softmax':
        # TODO: for out_dtype == 'float16' and epilogue == 'softmax', it will
        # fail with the following error: 'llvm.fmul' op requires the same type
        # for all operands and results
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         effective_in_dtype,
                         out_dtype,
                         COL_A=col_a, COL_B=col_b,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps,
                         matrix_instr_nonkdim=non_k_dim)
    # torch result
    if in_dtype == 'int8':
        z_ref = np.matmul(x.astype(np.float32),
                          y.astype(np.float32())).astype(np.int32)
    else:
        z_ref = np.matmul(x, y)

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
        z_ref = np.matmul(z_ref, w)
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    if in_dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    elif out_dtype == tl.float16:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    elif in_dtype == 'bfloat16':
        # added atol, to loose precision for bfloat16xbfloat16->float32 case
        # bfloat16 has less fraction bits than float16
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        # added atol, to loose precision for float16xfloat16->float32 case
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    if torch.version.hip is not None:
        backend = triton.common.backend.get_backend("hip")
        if backend.get_matrix_core_version() > 0:
            ttgir = pgm.asm['ttgir']
            if non_k_dim == 16:
                assert "#triton_gpu.mfma<{nonKDim = 16" in ttgir
                assert "#triton_gpu.mfma<{nonKDim = 32" not in ttgir
            elif non_k_dim == 32:
                assert "#triton_gpu.mfma<{nonKDim = 32" in ttgir
                assert "#triton_gpu.mfma<{nonKDim = 16" not in ttgir
        gcn = pgm.asm['amdgcn']
        if backend.get_matrix_core_version() == 3 and effective_in_dtype == tl.float8e5b16:
            assert "v_mfma_f32_32x32x16_bf8_bf8" in gcn or "v_mfma_f32_16x16x32_bf8_bf8" in gcn
        if backend.get_matrix_core_version() == 3 and effective_in_dtype == tl.float8e4b8:
            assert "v_mfma_f32_32x32x16_fp8_fp8" in gcn or "v_mfma_f32_16x16x32_fp8_fp8" in gcn
        return
    # make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    if (K > 16 or N > 16 or M > 16) and (M * N // (num_warps * 32) >= 4):
        # XXX: skip small sizes because they are not vectorized
        assert 'ld.global.v4' in ptx
        assert 'st.global.v4' in ptx
    if in_dtype == 'float32' and allow_tf32:
        assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
    elif in_dtype == 'float32' and allow_tf32:
        assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' not in ptx
    elif in_dtype == 'int8':
        assert 'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx
    elif out_dtype == tl.float16:
        assert 'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16' in ptx


@pytest.mark.parametrize("dtype_str", int_dtypes + float_dtypes + ['bfloat16'])
def test_full(dtype_str):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel_static(out):
        a = GENERATE_TEST_HERE
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    @triton.jit
    def kernel_dynamic(out, val, dtype: tl.constexpr):
        a = tl.full((128,), val, dtype)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    kernel_static_patched = patch_kernel(kernel_static, {'GENERATE_TEST_HERE': f"tl.full((128,), 2, tl.{dtype_str})"})
    out_static = torch.zeros((128), dtype=dtype, device="cuda")
    kernel_static_patched[(1,)](out_static)
    out_dynamic = torch.zeros((128), dtype=dtype, device="cuda")
    kernel_dynamic[(1,)](out_dynamic, 2, getattr(triton.language, dtype_str))
    assert torch.all(out_static == 2)
    assert torch.all(out_dynamic == 2)


# TODO: uncomment once DotOperandEncoding::getElemsPerThread is implemented
# @pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
# def test_dot_without_load(dtype_str):
#     @triton.jit
#     def _kernel(out):
#         a = GENERATE_TEST_HERE
#         b = GENERATE_TEST_HERE
#         c = tl.dot(a, b)
#         out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
#         tl.store(out_ptr, c)

#     kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
#     a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     out_ref = torch.matmul(a, b)
#     out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device="cuda")
#     kernel[(1,)](out)
#     assert torch.all(out == out_ref)

# ---------------
# test gemm
# ---------------

# Simplified matrix multiplication kernel
# Checks Matrix multiplication consists of several blocks is optimized correctly

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K)).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K)).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N)).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N)).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]


@pytest.mark.parametrize('SIZE_M,SIZE_N,SIZE_K,NUM_WARPS,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,NUM_STAGES', [
    [64, 32, 128, 4, 64, 32, 64, 0],
    [64, 32, 128, 4, 64, 32, 64, 2]
])
def test_gemm(SIZE_M, SIZE_N, SIZE_K, NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES):
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)
    grid = lambda META: (1, )
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                        num_warps=NUM_WARPS,
                        num_stages=NUM_STAGES)
    golden = torch.matmul(a, b)

    # It's not easy to get a proper error threshold in different size
    # Here the gemm calculation is padded to a different size in order to get
    # a variant version of the golden result. And the error between golden and
    # golden_variant provide reference on selecting the proper rtol / atol.
    golden_variant = get_variant_golden(a, b)
    golden_diff = golden - golden_variant
    golden_abs_err = torch.max(torch.abs(golden_diff)).item()
    golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()

    triton.testing.assert_close(c, golden.to(torch.float32), rtol=max(1e-4, 1.5 * golden_rel_err), atol=max(1e-4, 1.5 * golden_abs_err))

# ---------------
# test arange
# ---------------


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
    np.testing.assert_allclose(to_numpy(z_tri), to_numpy(z_ref))

# ---------------
# test load
# ---------------


@pytest.mark.parametrize("dtype_str, size, size_diff", [(dtype_str, size, size_diff) for dtype_str in torch_dtypes for size in [128, 512] for size_diff in [0, 1, 2, 3, 4]])
def test_masked_load(dtype_str, size, size_diff, device='cuda'):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype)  # bfloat16 on cc < 80 will not be tested

    input_size = size - size_diff
    output_size = size
    if dtype_str == 'bool':
        input = torch.randint(0, 2, (input_size,), dtype=dtype, device=device)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        input = torch.randint(0, 127, (input_size,), dtype=dtype, device=device)
    else:
        input = torch.rand(input_size, dtype=dtype, device=device)
    output = torch.zeros((output_size,), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in_ptr, out_ptr, in_size: tl.constexpr, out_size: tl.constexpr):
        in_offsets = tl.arange(0, out_size)
        # Load inputs.
        x = GENERATE_TEST_HERE
        # Store output
        output_offsets = tl.arange(0, out_size)
        tl.store(out_ptr + output_offsets, x)

    mask_str = "mask=in_offsets < in_size, other=1" if size_diff > 0 else "None"
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.load(in_ptr + in_offsets, {mask_str})"})
    kernel[(1,)](input, output, input_size, output_size)

    reference_out = torch.cat((input, torch.ones((size_diff,), dtype=dtype, device=device)))
    torch.testing.assert_allclose(output, reference_out)

# Testing masked loads with an intermate copy to shared memory run.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype, device='cuda'):
    check_type_supported(dtype)  # bfloat16 on cc < 80 will not be tested
    # skip bfloat on ROCM
    if torch.version.hip is not None and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
        pytest.skip("test_masked_load_shared_memory[bfloat16] is only supported on AMDGPU")

    M = 32
    N = 32
    K = 16

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr,
                in_stride, in2_stride, out_stride,
                in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
        in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < M * K)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < K * N)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w, out_dtype=tl.float32)

        # Store output
        output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < M * N)

    pgm = _kernel[(1,)](in1, in2, out,
                        in1.stride()[0],
                        in2.stride()[0],
                        out.stride()[0],
                        in1.numel(),
                        in2.numel(),
                        out.numel(),
                        M=M, N=N, K=K)

    reference_out = torch.matmul(in1, in2)
    torch.testing.assert_allclose(out, reference_out, atol=1e-2, rtol=0)


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
# test default
# ---------------
# TODO: can't be local to test_default


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

# ---------------
# test noop
# ----------------


def test_noop(device='cuda'):
    @triton.jit
    def kernel(x):
        pass
    x = to_triton(numpy_random((1,), dtype_str='int32'), device=device)
    kernel[(1, )](x)


@pytest.mark.parametrize("device", ['cuda', 'cpu', 'cpu_pinned'])
def test_pointer_arguments(device):
    @triton.jit
    def kernel(x):
        pass
    pin_memory = 'pinned' in device
    x = torch.empty(1024, device=device.split('_')[0], pin_memory=pin_memory)
    if device == "cpu":
        with pytest.raises(ValueError):
            kernel[(1,)](x)
    else:
        kernel[(1, )](x)


@pytest.mark.parametrize("value, value_type", [
    (-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
    (2**31, 'i64'), (2**32 - 1, 'i64'), (2**32, 'i64'), (2**63 - 1, 'i64'),
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

# --------------------
# value specialization
# --------------------


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


# ----------------
# test constexpr
# ----------------

@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '<', '>', '<<', '>>', '&', '^', '|'])
@pytest.mark.parametrize("is_lhs_constexpr", [False, True])
@pytest.mark.parametrize("is_rhs_constexpr", [True, False])
def test_bin_op_constexpr(op, is_lhs_constexpr, is_rhs_constexpr):

    @triton.jit
    def kernel(Z, X, Y):
        x = tl.load(X)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z, z)

    if op in ['<<', '>>', '&', '^', '|']:  # int op
        x_str = "3" if is_lhs_constexpr else "x"
        y_str = "4" if is_rhs_constexpr else "y"
        x = numpy_random((1,), dtype_str="int32")
        y = numpy_random((1,), dtype_str="int32")
    else:
        x_str = "3.14" if is_lhs_constexpr else "x"
        y_str = "4.13" if is_rhs_constexpr else "y"
        x = numpy_random((1,), dtype_str="float32")
        y = numpy_random((1,), dtype_str="float32")
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{x_str} {op} {y_str}"})
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

# -------------
# test call
# -------------


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

# -------------
# test if
# -------------

# TODO(Keren): if_exp_dynamic


@pytest.mark.parametrize("if_type", ["if", "if_and_dynamic", "if_exp_static", "if_and_static"])
def test_if(if_type, device):

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret, IfType: tl.constexpr, BoolVar: tl.constexpr, StaticVaue: tl.constexpr):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if IfType == "if":
            if pid % 2 == 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_dynamic":
            tl.store(Ret, tl.load(XTrue)) if pid % 2 == 0 else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_static":
            tl.store(Ret, tl.load(XTrue)) if BoolVar else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_dynamic":
            if BoolVar and pid % 2 == 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_static":
            if StaticVaue != 0 and StaticVaue != 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device=device)
    x_true = torch.tensor([3.14], dtype=torch.float32, device=device)
    x_false = torch.tensor([1.51], dtype=torch.float32, device=device)
    ret = torch.zeros(1, dtype=torch.float32, device=device)

    kernel[(1,)](cond, x_true, x_false, ret, if_type, True, 1)
    assert torch.equal(ret, x_true)


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

# -------------
# test extern
# -------------


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('int32', 'math.ffs', ''),
                          ('float32', 'math.log2', ''),
                          ('float32', 'math.scalbn', ''),
                          ('float32', 'math.pow', tl.math.libdevice_path()),
                          ('float64', 'math.pow_dtype', tl.math.libdevice_path()),
                          ('float64', 'math.norm4d', '')])
def test_math_tensor(dtype_str, expr, lib_path):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)

    if expr == 'math.log2':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.broadcast_to(tl.{expr}(5.0), x.shape)'})
        y_ref = np.log2(5.0)
    elif expr == 'math.ffs':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x)'})
        y_ref = np.zeros(shape, dtype=x.dtype)
        for i in range(shape[0]):
            y_ref[i] = (int(x[i]) & int(-x[i])).bit_length()
    elif expr == 'math.scalbn':
        pytest.skip("math.scalbn has mismatch issues")
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, 2)'})
        y_ref = x * pow(2, 2)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.math.pow(x, 0.5)'})
        y_ref = np.power(x, 0.5)
    elif expr == 'math.pow':
        # numpy does not allow negative factors in power, so we use abs()
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, x)'})
        y_ref = np.power(x, x)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, 0.5)'})
        y_ref = np.power(x, 0.5)
    elif expr == 'math.norm4d':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, x, x, x)'})
        y_ref = np.sqrt(4 * np.power(x, 2))

    x_tri = to_triton(x)
    # triton result
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    if expr == 'math.ffs':
        np.testing.assert_equal(y_ref, to_numpy(y_tri))
    else:
        np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('float32', 'math.pow', ''),
                          ('float64', 'math.pow_dtype', ''),
                          ('float64', 'math.pow', tl.math.libdevice_path())])
def test_math_scalar(dtype_str, expr, lib_path):

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
    if expr == 'math.pow':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, x)'})
        y_ref[:] = np.power(x, x)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, 0.5)'})
        y_ref[:] = np.power(x, 0.5)

    # triton result
    x_tri = to_triton(x)[0].item()
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device='cuda')
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path})
    # compare
    np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)

# -----------------------
# test control flow
# -----------------------


@pytest.mark.parametrize("lo, hi, iv", [(2**35, 2**35 + 20, 1), (2**35, 2**35 + 20, 2), (2**35, 2**35 + 20, 3),
                                        (15, -16, -1), (15, -16, -2), (15, -16, -3),
                                        (-18, -22, -1), (22, 18, -1)])
def test_for_iv(lo, hi, iv):

    @triton.jit
    def kernel(Out, lo, hi, iv: tl.constexpr):
        acc = 0
        acc = acc.to(tl.int64)
        for i in range(lo, hi, iv):
            acc += i
        tl.store(Out, acc)

    lo = 2**35
    hi = 2**35 + 20
    out = to_triton(np.zeros((1,), dtype=np.int64), device='cuda')
    kernel[(1,)](out, lo, hi, iv)
    assert out[0] == sum(range(lo, hi, iv))


def test_if_else():

    @triton.jit
    def kernel(Cond, TrueVal, FalseVal, Out):
        if tl.load(Cond):
            val = tl.load(TrueVal)
        else:
            val = tl.load(FalseVal)
        tl.store(Out, val)

    out = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    true_val = to_triton(np.full((1,), 1, dtype=np.int32), device='cuda')
    false_val = to_triton(np.full((1,), 2, dtype=np.int32), device='cuda')
    cond = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    # True
    cond[0] = True
    kernel[(1,)](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == true_val[0]
    # False
    cond[0] = False
    kernel[(1,)](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == false_val[0]


def test_if_return():

    @triton.jit
    def kernel(ExitEarly, Out):
        if tl.load(ExitEarly):
            tl.store(Out, 0)
            return
        tl.store(Out, 1)

    out = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    exit_early = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    # exit early path taken
    exit_early[0] = 1
    kernel[(1,)](exit_early, out)
    assert to_numpy(out)[0] == 0
    # exit early path not taken
    exit_early[0] = 0
    kernel[(1,)](exit_early, out)
    assert to_numpy(out)[0] == 1


@pytest.mark.parametrize("_cond1", [True, False])
@pytest.mark.parametrize("_cond2", [True, False])
@pytest.mark.parametrize("_cond3", [True, False])
def test_nested_if_else_return(_cond1, _cond2, _cond3):

    @triton.jit
    def kernel(Cond1, Cond2, Cond3, Val1, Val2, Val3, Out):
        val = 0
        if tl.load(Cond1):
            if tl.load(Cond2):
                val = tl.load(Val1)
            else:
                return
        else:
            if tl.load(Cond3):
                val = tl.load(Val2)
            else:
                val = tl.load(Val3)
        tl.store(Out, val)

    out = to_triton(np.full((1,), -1, dtype=np.int32), device='cuda')
    cond1 = to_triton(np.full((1,), _cond1, dtype=np.int32), device='cuda')
    cond2 = to_triton(np.full((1,), _cond2, dtype=np.int32), device='cuda')
    cond3 = to_triton(np.full((1,), _cond3, dtype=np.int32), device='cuda')
    val1 = to_triton(np.full((1,), 1, dtype=np.int32), device='cuda')
    val2 = to_triton(np.full((1,), 2, dtype=np.int32), device='cuda')
    val3 = to_triton(np.full((1,), 3, dtype=np.int32), device='cuda')
    kernel[(1,)](cond1, cond2, cond3, val1, val2, val3, out)
    targets = {
        (True, True, True): val1[0],
        (True, True, False): val1[0],
        (True, False, True): out[0],
        (True, False, False): out[0],
        (False, True, True): val2[0],
        (False, True, False): val3[0],
        (False, False, True): val2[0],
        (False, False, False): val3[0],
    }
    assert out[0] == targets[(_cond1, _cond2, _cond3)]


def test_while():

    @triton.jit
    def kernel(InitI, Bound, CutOff, OutI, OutJ):
        init_i = tl.load(InitI)
        curr_i = init_i
        j = 0
        while curr_i == init_i and j < tl.load(Bound):
            curr_i = curr_i + (j == tl.load(CutOff))
            j += 1
        tl.store(OutI, curr_i)
        tl.store(OutJ, j)

    out_i = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    out_j = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
    init_i = to_triton(np.full((1,), 1, dtype=np.int32), device='cuda')
    bound = to_triton(np.full((1,), 10, dtype=np.int32), device='cuda')
    cut_off = to_triton(np.full((1,), 5, dtype=np.int32), device='cuda')
    kernel[(1,)](init_i, bound, cut_off, out_i, out_j)
    assert out_i[0] == init_i[0] + 1
    assert out_j[0] == cut_off[0] + 1

# def test_for_if():

#     @triton.jit
#     def kernel(bound, cutoff, M, N):
#         m = 0
#         n = 0
#         for i in range(bound):
#             if i > cutoff:
#                 m = m + 1
#             else:
#                 n = n + 1
#         tl.store(M, m)
#         tl.store(N, n)

#     m = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
#     n = to_triton(np.zeros((1,), dtype=np.int32), device='cuda')
#     kernel[(1,)](10, 7, m, n)
#     print(m[0])
#     print(n[0])


# -----------------------
# test layout conversions
# -----------------------
# TODO: backend should be tested separately

class MmaLayout:
    def __init__(self, version, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape):
        self.version = version
        self.warps_per_cta = str(warps_per_cta)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)
        self.instr_shape = str(instr_shape)

    def __str__(self):
        return f"#{GPU_DIALECT}.mma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA={self.warps_per_cta}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}, instrShape={self.instr_shape}}}>"


class MfmaLayout:
    def __init__(self, non_k_dim, warps_per_cta, is_transposed, ctas_per_cga, cta_split_num, cta_order):
        self.non_k_dim = str(non_k_dim)
        self.warps_per_cta = str(warps_per_cta)
        self.is_transposed = str(is_transposed).lower()
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)

    def __str__(self):
        return f"#{GPU_DIALECT}.mfma<{{nonKDim = {self.non_k_dim}, warpsPerCTA = {self.warps_per_cta}, isTransposed = {self.is_transposed}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class BlockedLayout:
    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga, cta_split_num, cta_order):
        self.sz_per_thread = str(size_per_thread)
        self.threads_per_warp = str(threads_per_warp)
        self.warps_per_cta = str(warps_per_cta)
        self.order = str(order)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class SharedLayout:
    def __init__(self, vec, per_phase, max_phase, order, ctas_per_cga, cta_split_num, cta_order):
        self.vec = str(vec)
        self.per_phase = str(per_phase)
        self.max_phase = str(max_phase)
        self.order = str(order)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)

    def __str__(self):
        return f"#{GPU_DIALECT}.shared<{{vec={self.vec}, perPhase={self.per_phase}, maxPhase={self.max_phase}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


@pytest.mark.parametrize("vec_size", [2, 4])
@pytest.mark.parametrize("swizzle", [True, False])
@pytest.mark.parametrize("transposeA", [True, False])
@pytest.mark.parametrize("transposeB", [True, False])
def test_dot_mfma_vector_load(vec_size, swizzle, transposeA, transposeB):
    # we can not do vector loads in this case
    if transposeA and not transposeB:
        pytest.skip()

    if triton.common.backend.get_backend("hip").get_matrix_core_version() == 0:
        pytest.skip("mfma is not available on hardware")

    # source code for following ttgir:
    # @triton.jit
    # def kernel(X, Y, Z):
    #     off_m = tl.arange(0, 32)
    #     off_n = tl.arange(0, 32)
    #     off_k = tl.arange(0, 32)
    #     Xs = X + off_m[:, None] * 32 + off_k[None, :]
    #     Ys = Y + off_k[:, None] * 32 + off_n[None, :]
    #     Zs = Z + off_m[:, None] * 32 + off_n[None, :]
    #     x = tl.load(Xs)
    #     y = tl.load(Ys)
    #     z = tl.dot(x, y, allow_tf32=False)
    #     tl.store(Zs, z)

    if swizzle:
        max_phase = 2
    else:
        max_phase = 1
        if vec_size != 4:
            pytest.skip()

    blocked = BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[8, 8], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1])
    shared_a = SharedLayout(vec=vec_size, per_phase=1, max_phase=max_phase, order=[0, 1] if transposeA else [1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1])
    shared_b = SharedLayout(vec=vec_size, per_phase=1, max_phase=max_phase, order=[0, 1] if transposeB else [1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1])
    mfma = MfmaLayout(non_k_dim=32, warps_per_cta=[4, 1], is_transposed=False, ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1])

    ir = f"""
#blocked = {blocked}
#shared1 = {shared_a}
#shared2 = {shared_b}
#mfma = {mfma}
""" + """
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @kernel_0d1d2d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mfma>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<32x1xi32, #blocked>
    %3 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %7 = tt.broadcast %4 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %8 = tt.broadcast %6 : (tensor<1x32xi32, #blocked>) -> tensor<32x32xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %10 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %2 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %12 = tt.broadcast %11 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %13 = tt.addptr %12, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %14 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %15 = tt.addptr %14, %2 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %16 = tt.broadcast %15 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %17 = tt.addptr %16, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %18 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16, #blocked>
    %19 = triton_gpu.convert_layout %18 : (tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #shared1>
    %20 = triton_gpu.convert_layout %19 : (tensor<32x32xf16, #shared1>) -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>>
    %21 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16, #blocked>
    %22 = triton_gpu.convert_layout %21 : (tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #shared2>
    %23 = triton_gpu.convert_layout %22 : (tensor<32x32xf16, #shared2>) -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth=4}>>
    %24 = tt.dot %20, %23, %cst {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth=4}>> -> tensor<32x32xf32, #mfma>
    %25 = triton_gpu.convert_layout %24 : (tensor<32x32xf32, #mfma>) -> tensor<32x32xf32, #blocked>
    %26 = arith.truncf %25 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    tt.store %17, %26 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf16, #blocked>
    tt.return
  }
}
"""
    shape = (32, 32)
    dtype = 'float16'
    x_np = numpy_random(shape, dtype_str=dtype)
    y_np = numpy_random(shape, dtype_str=dtype)
    z_np = np.dot(x_np, y_np)

    x_tri = to_triton(x_np)
    y_tri = to_triton(y_np)
    z_tri = torch.empty_like(x_tri)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        backend = triton.common.backend.get_backend("hip")
        capabilities = backend.get_architecture_descriptor()
        kernel = triton.compile(f.name, device_type="hip", cc=capabilities)

    import triton.language.semantic as sem
    # if torch.version.hip is not None and sem.gpu_matrix_core_version() > 0:
    if torch.version.hip is not None and backend.get_matrix_core_version() > 0:
        kernel[(1, 1, 1)](x_tri, y_tri, z_tri)
        np.testing.assert_allclose(z_np, to_numpy(z_tri), rtol=0.01, atol=1e-3)

    assert(f"load <{vec_size} x half>, ptr addrspace(3)" in kernel.asm['llir'])


def _get_warp_size():
    if torch.version.hip is None:
        return 32  # CUDA_DEFAULT_WARP_SIZE
    return _triton.get_warp_size()


if is_hip():
    layouts = [
        # MmaLayout(version=1, warps_per_cta=[1, 4]),
        # MmaLayout(version=(2, 0), warps_per_cta=[1, 4]),
        # MmaLayout(version=1, warps_per_cta=[4, 1]),
        # MmaLayout(version=(2, 0), warps_per_cta=[4, 1]),
        BlockedLayout([1, 2], [2, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([2, 2], [4, 16], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([1, 1], [1, 64], [2, 2], [1, 0],[1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [16, 4], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [8, 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([1, 1], [32, 2], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [1, 64], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1])
    ]
else:
    layouts = [
        # MmaLayout(version=1, warps_per_cta=[1, 4]),
        MmaLayout(version=(2, 0), warps_per_cta=[1, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1]),
        # MmaLayout(version=1, warps_per_cta=[4, 1]),
        MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1]),
        BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([2, 2], [4, 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [16, 2], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [8, 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [4, 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([1, 1], [16, 2], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 2], [1, 32], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1])
    ]


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("dst_layout", layouts)
def test_convert2d(dtype, shape, src_layout, dst_layout, device='cuda'):
    if str(src_layout) == str(dst_layout):
        pytest.skip()
    if 'mma' in str(src_layout) and 'mma' in str(dst_layout):
        pytest.skip()

    ir = f"""
#src = {src_layout}
#dst = {dst_layout}
""" + """
module attributes {"triton_gpu.num-ctas" = 1, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = """ + str(_get_warp_size()) + """ : i32} {
  tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<128> : tensor<128x1xi32, #src>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>
    %2 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #src>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>) -> tensor<128x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<128x1xi32, #src>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>) -> tensor<1x128xi32, #src>
    %7 = tt.broadcast %6 : (tensor<1x128xi32, #src>) -> tensor<128x128xi32, #src>
    %8 = tt.broadcast %5 : (tensor<128x1xi32, #src>) -> tensor<128x128xi32, #src>
    %9 = arith.addi %8, %7 : tensor<128x128xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<128x128x!tt.ptr<f16>, #src>, tensor<128x128xi32, #src>
    %11 = tt.load %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #src>
    %3 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #dst>
    %12 = triton_gpu.convert_layout %9 : (tensor<128x128xi32, #src>) -> tensor<128x128xi32, #dst>
    %13 = triton_gpu.convert_layout %11 : (tensor<128x128xf16, #src>) -> tensor<128x128xf16, #dst>
    %14 = tt.addptr %3, %12 : tensor<128x128x!tt.ptr<f16>, #dst>, tensor<128x128xi32, #dst>
    tt.store %14, %13 : tensor<128x128xf16, #dst>
    tt.return
  }
}
"""

    x = to_triton(numpy_random(shape, dtype_str=dtype))
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    assert torch.equal(z, x)


if torch.version.hip is not None and _get_warp_size() == 64:
    layouts = [
        MfmaLayout(non_k_dim=32, warps_per_cta=[4, 1], is_transposed=True, ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0]),
        MfmaLayout(non_k_dim=32, warps_per_cta=[2, 2], is_transposed=False, ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0]),
    ]
    shapes = [[128, 32], [128, 128], [32, 128], [64, 64]]
else:
    layouts = [
        BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([1, 4], [8, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
        BlockedLayout([4, 4], [2, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
        MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
        MmaLayout(version=(2, 0), warps_per_cta=[2, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
        MmaLayout(version=(3, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 16, 16]),
    ]
    shapes = [[128, 16], [128, 128], [32, 128]]


@pytest.mark.parametrize("M, N", shapes)
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_layouts(M, N, src_layout, axis, device='cuda'):
    if torch.version.hip is not None and _get_warp_size() == 64:
        if src_layout.is_transposed and axis == 0:
            pytest.skip("Reduce along axis 0 is not supported in transposed mfma layout")
    rdims_2d = f"1x{N}" if axis == 0 else f"{M}x1"
    rdims_1d = f"{N}" if axis == 0 else f"{M}"
    store_range = "%7" if axis == 0 else "%1"
    blocked = BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1])
    ir = f"""
    #blocked = {blocked}
    #src = {src_layout}
    module attributes {{"triton_gpu.num-ctas" = 1, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = {_get_warp_size()} : i32}} {{
    tt.func public @kernel_0d1d2c3d4c(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}) {{
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>) -> tensor<{M}x1xi32, #blocked>
        %2 = tt.splat %arg1 : (i32) -> tensor<{M}x1xi32, #blocked>
        %3 = arith.muli %1, %2 : tensor<{M}x1xi32, #blocked>
        %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<{M}x1x!tt.ptr<f32>, #blocked>
        %5 = tt.addptr %4, %3 : tensor<{M}x1x!tt.ptr<f32>, #blocked>, tensor<{M}x1xi32, #blocked>
        %6 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>
        %7 = tt.expand_dims %6 {{axis = 0 : i32}} : (tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>) -> tensor<1x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : (tensor<{M}x1x!tt.ptr<f32>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<f32>, #blocked>
        %9 = tt.broadcast %7 : (tensor<1x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
        %10 = tt.addptr %8, %9 : tensor<{M}x{N}x!tt.ptr<f32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<{rdims_2d}x!tt.ptr<f32>, #blocked>
        %12 = tt.addptr %11, {store_range} : tensor<{rdims_2d}x!tt.ptr<f32>, #blocked>, tensor<{rdims_2d}xi32, #blocked>
        %13 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}xf32, #blocked>
        %14 = triton_gpu.convert_layout %13 : (tensor<{M}x{N}xf32, #blocked>) -> tensor<{M}x{N}xf32, #src>
        %15 = "tt.reduce"(%14) ({{
        ^bb0(%arg3: f32, %arg4: f32):
          %16 = "arith.cmpf"(%arg3, %arg4) {{predicate = 2 : i64}} : (f32, f32) -> i1
          %17 = arith.select %16, %arg3, %arg4 : f32
          tt.reduce.return %17 : f32
        }}) {{axis = {axis} : i32}} : (tensor<{M}x{N}xf32, #src>) -> tensor<{rdims_1d}xf32, #triton_gpu.slice<{{dim = {axis}, parent = #src}}>>
        %18 = triton_gpu.convert_layout %15 : (tensor<{rdims_1d}xf32, #triton_gpu.slice<{{dim = {axis}, parent = #src}}>>) -> tensor<{rdims_1d}xf32, #triton_gpu.slice<{{dim = {axis}, parent = #blocked}}>>
        %19 = tt.expand_dims %18 {{axis = {axis} : i32}} : (tensor<{rdims_1d}xf32, #triton_gpu.slice<{{dim = {axis}, parent = #blocked}}>>) -> tensor<{rdims_2d}xf32, #blocked>
        tt.store %12, %19 {{cache = 1 : i32, evict = 1 : i32}} : tensor<{rdims_2d}xf32, #blocked>
        tt.return
    }}
    }}
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, N)).astype('float32')
    x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')

    if axis == 0:
        z = np.zeros((1, N)).astype('float32')
    else:
        z = np.zeros((M, 1)).astype('float32')

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)
    pgm = kernel[(1, 1, 4)](x_tri, x_tri.stride(0), z_tri)

    z_ref = np.max(x, axis=axis, keepdims=True)

    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


scan_layouts = [
    BlockedLayout([1, 4], [4, 16], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, 16], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, 16], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),

    BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, 16], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, 16], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[32, 32], [32, 64], [64, 32]])
@pytest.mark.parametrize("src_layout", scan_layouts)
@pytest.mark.parametrize("axis", [0, 1])
def test_scan_layouts(M, N, src_layout, axis, device):
    ir = f"""
    #blocked = {src_layout}
    module attributes {{"triton_gpu.num-ctas" = 1, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32}} {{
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
      %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
      %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>
      %1 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>) -> tensor<{M}x1xi32, #blocked>
      %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #blocked>
      %3 = tt.splat %arg0 : (!tt.ptr<i32>) -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %4 = tt.addptr %3, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>
      %6 = tt.expand_dims %5 {{axis = 0 : i32}} : (tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>) -> tensor<1x{N}xi32, #blocked>
      %7 = tt.broadcast %4 : (tensor<{M}x1x!tt.ptr<i32>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %8 = tt.broadcast %6 : (tensor<1x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %9 = tt.addptr %7, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      %10 = tt.load %9 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}xi32, #blocked>
      %11 = "tt.scan"(%10) <{{axis = {axis} : i32}}> ({{
      ^bb0(%arg2: i32, %arg3: i32):
        %16 = arith.addi %arg2, %arg3 : i32
        tt.scan.return %16 : i32
      }}) : (tensor<{M}x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %12 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %13 = tt.addptr %12, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %14 = tt.broadcast %13 : (tensor<{M}x1x!tt.ptr<i32>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %15 = tt.addptr %14, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      tt.store %15, %11 {{cache = 1 : i32, evict = 1 : i32}} : tensor<{M}x{N}xi32, #blocked>
      tt.return
    }}
    }}
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    rs = RandomState(17)
    x = rs.randint(-100, 100, (M, N)).astype('int32')

    z = np.zeros((M, N)).astype('int32')
    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)

    z_ref = np.cumsum(x, axis=axis)

    np.testing.assert_equal(z_ref, z_tri.cpu().numpy())


@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", [MfmaLayout(non_k_dim=32, warps_per_cta=[2, 1], is_transposed=False, ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0]),
                                        MfmaLayout(non_k_dim=32, warps_per_cta=[4, 1], is_transposed=True, ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])])
@pytest.mark.parametrize("dst_layout", [BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0], [1, 1], [1, 1], [0, 1])])
def test_make_range(dtype, shape, src_layout, dst_layout, device='cuda'):
    ir = f"""
#src = {src_layout}
#dst = {dst_layout}
""" + """
module attributes {"triton_gpu.num-ctas" = 1, "triton_gpu.num-warps" = """ + str(128 // _get_warp_size()) + """ : i32, "triton_gpu.threads-per-warp" = """ + str(_get_warp_size()) + """ : i32} {
  tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<64> : tensor<64x1xi32, #src>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>
    %2 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #dst>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>) -> tensor<64x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<64x1xi32, #src>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>) -> tensor<1x64xi32, #src>
    %7 = tt.broadcast %6 : (tensor<1x64xi32, #src>) -> tensor<64x64xi32, #src>
    %8 = tt.broadcast %5 : (tensor<64x1xi32, #src>) -> tensor<64x64xi32, #src>
    %9 = arith.addi %8, %7 : tensor<64x64xi32, #src>
    %33 = triton_gpu.convert_layout %9 : (tensor<64x64xi32, #src>) ->  tensor<64x64xi32, #dst>
    %10 = tt.addptr %2, %33 : tensor<64x64x!tt.ptr<f16>, #dst>, tensor<64x64xi32, #dst>
    %11 = tt.load %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf16, #dst>
    %3 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #dst>
    %12 = triton_gpu.convert_layout %9 : (tensor<64x64xi32, #src>) -> tensor<64x64xi32, #dst>
    %14 = tt.addptr %3, %12 : tensor<64x64x!tt.ptr<f16>, #dst>, tensor<64x64xi32, #dst>
    tt.store %14, %11 : tensor<64x64xf16, #dst>
    tt.return
  }
}
"""
    x = to_triton(numpy_random(shape, dtype_str=dtype))
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    assert torch.equal(z, x)


def test_load_scalar_with_mask():
    @triton.jit
    def kernel(Input, Index, Out, N: int):
        index = tl.load(Index)
        scalar = tl.load(Input + index, mask=index < N, other=0)
        tl.store(Out, scalar, mask=index < N)
    Index = torch.tensor([0], dtype=torch.int32, device='cuda')
    Input = torch.tensor([0], dtype=torch.int32, device='cuda')
    Out = torch.empty_like(Index, device='cuda')
    kernel[(1,)](Input, Index, Out, Index.numel())
    assert Out.data[0] == 0
