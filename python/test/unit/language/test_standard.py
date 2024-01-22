import triton
import pytest
import torch
import triton.language as tl
import numpy as np

from test_core import _test_binary, int_dtypes, uint_dtypes, float_dtypes, is_hip, numpy_random, check_type_supported, to_triton, to_numpy, RandomState, patch_kernel


# ---------------
# test maximum/minimum ops
# ---------------

# TODO: Tests with unsigned integers failed at compilation stage.
@pytest.mark.parametrize("dtype", int_dtypes + uint_dtypes + float_dtypes + ["bfloat16"])
@pytest.mark.parametrize("op", ["maximum", "minimum"])
def test_maximum_minium(dtype, op):
    expr = f'tl.{op}(x, y)'
    numpy_expr = f'np.{op}(x, y)'
    _test_binary(dtype, dtype, expr, numpy_expr)


# ---------------
# test sort op
# ---------------

@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32'])
def test_sort(M, N, descending, dtype_str, device):
    if is_hip():
        pytest.skip(
            'test_propagate_nan for HIP currently broken in https://github.com/openai/triton. Use https://github.com/ROCmSoftwarePlatform/triton'
        )

    @triton.jit
    def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.sort(x, descending=descending)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to("cuda")
    y = torch.sort(x, descending=descending)[0]
    z = torch.empty_like(x)
    sort_kernel[(1, )](x, z, N, M, descending, num_warps=8)
    assert (y == z).all(), (y, z)


# ---------------
# test flip op
# ---------------

@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32'])
def test_flip(M, N, dtype_str, device):

    @triton.jit
    def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.flip(x)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to("cuda")
    y = torch.flip(x, (1, ))
    z = torch.empty_like(x)
    flip_kernel[(1, )](x, z, N, M, num_warps=8)
    assert (y == z).all(), (y, z)


# ---------------
# test scan ops
# ---------------

scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [(op, type, shape, axis, num_warps)
                for num_warps in [4, 16]
                for type in ['int32', 'float32']
                for axis in [1, 0]
                for shape in scan2d_shapes
                for op in ['cumsum', 'cumprod', 'get_first_element', 'linear_recurrence']]
negative_config = [('cumsum', 'float32', (32, 32), -1, 4)]


@triton.jit
# trivial associative but not commutative function
def get_first_element(a, b):
    return a


# Compute x_i = a_i * x_{i-1} + b_i
@triton.jit
def linear_recurrence(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2


@pytest.mark.parametrize("op, dtype_str, shape, axis, num_warps", scan_configs + negative_config)
def test_scan2d(op, dtype_str, shape, axis, num_warps, device):
    if is_hip():
        pytest.skip("test_scan2d is not supported in HIP")
    check_type_supported(dtype_str, device)

    # triton kernel
    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        y = tl.load(Y + range_m[:, None] * BLOCK_N + range_n[None, :])
        GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    if op == 'cumsum' or op == 'cumprod':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'z = tl.{op}(x, axis={axis})'})
    elif op == 'get_first_element':
        kernel = patch_kernel(kernel,
                              {'GENERATE_TEST_HERE': f'z = tl.associative_scan(x, axis={axis}, combine_fn={op})'})
    else:
        assert op == 'linear_recurrence'
        kernel = patch_kernel(
            kernel, {'GENERATE_TEST_HERE': f'_, z = tl.associative_scan((x, y), axis={axis}, combine_fn={op})'})
    # input
    rs = RandomState(17)
    if op == 'linear_recurrence' and dtype_str in int_dtypes:
        # If the numbers are too large the op will overflow
        # We sample numbers in -1, 0, 1
        x = rs.randint(-1, 2, shape, dtype=dtype_str)
        y = rs.randint(-1, 2, shape, dtype=dtype_str)
    else:
        x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
        # y is just used in linear_recurrence
        y = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    if op == 'cumsum' or op == 'cumprod':
        numpy_op = {'cumsum': np.cumsum, 'cumprod': np.cumprod}[op]
        z_dtype_str = dtype_str
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    elif op == 'linear_recurrence':
        # Simplify to the axis=1 case
        x_ref = x.T if axis == 0 else x
        y_ref = y.T if axis == 0 else y
        result = []
        for x_refi, y_refi in zip(x_ref, y_ref):
            li = []
            acc = 0
            for xi, yi in zip(x_refi, y_refi):
                acc = xi * acc + yi
                li.append(acc)
            result.append(li)
        z_ref = np.array(result)
        if axis == 0:
            z_ref = z_ref.T
    else:
        assert op == 'get_first_element'
        z_ref = x
        if axis == 0:
            z_ref[1:] = x[0]
        else:
            z_ref[:, 1:] = x[:, 0:1]
    # triton result
    z_tri = to_triton(z, device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)
    z_tri = to_numpy(z_tri)
    # compare
    if dtype_str == 'float32':
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)
