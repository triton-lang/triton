import warnings

import triton
import pytest
import torch
import triton.language as tl

from test_core import _test_binary, int_dtypes, uint_dtypes, float_dtypes, numpy_random
from triton._internal_testing import is_interpreter
from triton.runtime.errors import InterpreterError

# ---------------
# test maximum/minimum ops
# ---------------


# TODO: Tests with unsigned integers failed at compilation stage.
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", int_dtypes + uint_dtypes + float_dtypes + ["bfloat16"])
@pytest.mark.parametrize("op", ["maximum", "minimum"])
def test_maximum_minium(dtype, op, device):
    expr = f'tl.{op}(x, y)'
    numpy_expr = f'np.{op}(x, y)'
    _test_binary(dtype, dtype, expr, numpy_expr, device=device)


# ---------------
# test sort op
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 1], [1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("k", [None, 1, 8])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
@pytest.mark.enable_warmup(min_capability=9)
def test_sort(M, N, k, descending, dtype_str, device):

    @triton.jit
    def sort_kernel(X, stride_xm, Z, stride_zm, M: tl.constexpr, N: tl.constexpr, k: tl.constexpr,
                    descending: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_x_n = tl.arange(0, N)
        offs_z_n = offs_x_n if k is None else tl.arange(0, k)
        offs_x = offs_m[:, None] * stride_xm + offs_x_n[None, :]
        x = tl.load(X + offs_x)
        if k is None or x.numel < k:
            z = tl.sort(x, descending=descending)
        else:
            z = tl.topk(x, k, descending=descending)
        offs_z = offs_m[:, None] * stride_zm + offs_z_n[None, :]
        tl.store(Z + offs_z, z)

    z_shape = (M, N if k is None else k)
    x = numpy_random((M, N), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    z = torch.empty(z_shape, dtype=x.dtype, device=x.device)
    if k is None or x.numel() < k:
        y = torch.sort(x, descending=descending)[0]
    else:
        y = torch.topk(x, k=k, largest=descending).values
    sort_kernel[(1, )](x, x.stride(0), z, z.stride(0), M, N, k, descending, num_warps=8)
    assert (y == z).all(), (y, z)


# ---------------
# test flip op
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N, K", [[1, 16, 64], [8, 2, 256], [32, 1, 2], [128, 8, 1]])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
@pytest.mark.parametrize("dim", [0, 1, 2, -2, None])
def test_flip(M, N, K, dtype_str, dim, device):

    @triton.jit
    def flip_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, dim: tl.constexpr):
        offx = tl.arange(0, M) * N * K
        offy = tl.arange(0, N) * K
        offz = tl.arange(0, K)
        off3d = offx[:, None, None] + offy[None, :, None] + offz[None, None, :]
        x = tl.load(X + off3d)
        x = tl.flip(x, dim)
        tl.store(Z + off3d, x)

    x = numpy_random((M, N, K), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.flip(x, (dim if dim is not None else -1, ))
    z = torch.empty_like(x, device=device)
    flip_kernel[(1, )](x, z, M, N, K, dim, num_warps=8)
    assert (y == z).all(), (y, z)


@pytest.mark.interpreter
def test_flip_inf(device):
    # Reproducer for https://github.com/triton-lang/triton/issues/5439

    @triton.jit
    def triton_flip_kernel(out_ptr, x_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        x = tl.load(x_ptr + pid * N + tl.arange(0, N))
        shape: tl.constexpr = (N // 2, 2)
        y = x.reshape(shape)
        y = tl.flip(y, dim=1).reshape(x.shape)
        tl.store(out_ptr + pid * N + tl.arange(0, N), y)

    x = torch.arange(0, 16, device=device).unsqueeze(0).float()
    x[:, -1] = float('inf')

    expect = x.reshape(-1, 8, 2).flip(-1).reshape(-1, 16)
    actual = torch.empty_like(x)
    triton_flip_kernel[(x.shape[0], )](actual, x, x.shape[1])

    torch.testing.assert_close(expect, actual)


@pytest.mark.interpreter
def test_ravel(device):

    @triton.jit
    def triton_ravel(out_ptr):
        a = tl.arange(0, 256)
        a = tl.reshape(a, (32, 8))
        a = tl.ravel(a)
        tl.store(out_ptr + tl.arange(0, 256), a)

    out = torch.empty((256, ), device=device, dtype=torch.int32)
    triton_ravel[(1, )](out)

    assert (out == torch.arange(0, 256, device=device)).all()


@pytest.mark.interpreter
@pytest.mark.parametrize("size_i, size_j, size_g", [[5, 7, 3]])
def test_swizzle2d(size_i, size_j, size_g, device):

    @triton.jit
    def swizzle2d_kernel(output, size_i, size_j, size_g):
        for i in tl.range(0, size_i, 1):
            for j in tl.range(0, size_j, 1):
                new_i, new_j = tl.swizzle2d(i, j, size_i, size_j, size_g)
                tl.store(output + new_i * size_j + new_j, i * size_j + j)

    output = torch.zeros(size_i, size_j).to(device)
    swizzle2d_kernel[(1, )](output, size_i, size_j, size_g)
    expected_order = torch.tensor([[0, 3, 6, 9, 12, 15, 18], [1, 4, 7, 10, 13, 16, 19], [2, 5, 8, 11, 14, 17, 20],
                                   [21, 23, 25, 27, 29, 31, 33], [22, 24, 26, 28, 30, 32, 34]]).to(device)
    assert (output == expected_order).all(), (output, expected_order)


# ---------------
# test softmax
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, dim, ieee_rounding", [((2, 2), 1, False), ((8, 64), -1, True), ((128, ), 0, False)])
def test_softmax(shape, dim, ieee_rounding, device):
    # Reproducer for https://github.com/triton-lang/triton/issues/11406, where
    # softmax normalized along the wrong axis for every dim but the default.

    @triton.jit
    def softmax_kernel(X, Z, numel: tl.constexpr, shape: tl.constexpr, dim: tl.constexpr, ieee_rounding: tl.constexpr):
        # X is contiguous, so its row-major flat offsets are just an arange.
        offs = tl.arange(0, numel).reshape(shape)
        x = tl.load(X + offs)
        if ieee_rounding:
            z = x.softmax(dim=dim, ieee_rounding=True)
        else:
            z = tl.softmax(x, dim=dim)
        tl.static_assert(z.shape == x.shape, "softmax must preserve the input shape")
        tl.store(Z + offs, z)

    x = numpy_random(shape, dtype_str='float32')
    x = torch.from_numpy(x).to(device)
    z = torch.empty_like(x)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*keep_dims.*")
        softmax_kernel[(1, )](x, z, x.numel(), shape, dim, ieee_rounding)
    torch.testing.assert_close(z, torch.softmax(x, dim=dim), rtol=1e-5, atol=1e-6)


@pytest.mark.interpreter
@pytest.mark.parametrize("member", [False, True])
def test_softmax_rejects_positional_keep_dims(member):

    @triton.jit
    def kernel(member: tl.constexpr):
        x = tl.full((2, 2), 1.0, tl.float32)
        if member:
            x.softmax(1, True)
        else:
            tl.softmax(x, 1, True)

    error = InterpreterError if is_interpreter() else triton.CompilationError
    with pytest.raises(error, match="positional argument"):
        kernel[(1, )](member)


@pytest.mark.interpreter
@pytest.mark.parametrize("keep_dims", [None, False, True])
@pytest.mark.parametrize("member", [False, True])
def test_softmax_keep_dims_deprecated(keep_dims, member, device, fresh_triton_cache):

    @triton.jit
    def kernel(X, Z, keep_dims: tl.constexpr, member: tl.constexpr):
        offs = tl.arange(0, 32).reshape((8, 4))
        x = tl.load(X + offs)
        if member:
            z = x.softmax(dim=1, keep_dims=keep_dims, ieee_rounding=True)
        else:
            z = tl.softmax(x, dim=1, keep_dims=keep_dims, ieee_rounding=True)
        tl.static_assert(z.shape == x.shape, "softmax must preserve the input shape")
        tl.store(Z + offs, z)

    x = torch.from_numpy(numpy_random((8, 4), dtype_str='float32')).to(device)
    z = torch.empty_like(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel[(1, )](x, z, keep_dims, member)
    if keep_dims is None:
        assert not caught
    else:
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert "keep_dims argument to tl.softmax is deprecated and ignored" in str(caught[0].message)
    torch.testing.assert_close(z, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, dim", [((1, 2, 4), 0), ((2, 1, 4), 1), ((2, 4, 1), 2)])
def test_squeeze(shape, dim, device):

    @triton.jit
    def triton_squeeze(out_ptr, dim: tl.constexpr, s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr):
        a = tl.arange(0, 8)
        a = tl.reshape(a, (s0, s1, s2))
        a = tl.squeeze(a, dim)
        a = tl.ravel(a)
        tl.store(out_ptr + tl.arange(0, 8), a)

    out = torch.empty((8, ), device=device, dtype=torch.int32)
    triton_squeeze[(1, )](out, dim, shape[0], shape[1], shape[2])

    expected = torch.arange(0, 8, device=device, dtype=torch.int32)
    expected = expected.reshape(shape).squeeze(dim).reshape(-1)
    assert (out == expected).all()


@pytest.mark.interpreter
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_unsqueeze(dim, device):

    @triton.jit
    def triton_unsqueeze(out_ptr, dim: tl.constexpr):
        a = tl.arange(0, 8)
        a = tl.reshape(a, (2, 4))
        a = tl.unsqueeze(a, dim)
        a = tl.ravel(a)
        tl.store(out_ptr + tl.arange(0, 8), a)

    out = torch.empty((8, ), device=device, dtype=torch.int32)
    triton_unsqueeze[(1, )](out, dim)

    expected = torch.arange(0, 8, device=device, dtype=torch.int32)
    expected = expected.reshape(2, 4).unsqueeze(dim).reshape(-1)
    assert (out == expected).all()


@pytest.mark.interpreter
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_max_propagate_nan(propagate_nan, device):

    @triton.jit
    def max_kernel(in_ptr, out_ptr, propagate_nan: tl.constexpr):
        a = tl.load(in_ptr + tl.arange(0, 8)[:, None] * 8 + tl.arange(0, 8)[None, :])
        a = tl.max(a, 0, propagate_nan=propagate_nan)
        tl.store(out_ptr + tl.arange(0, 8), a)

    a = torch.randn((8, 8), device=device)
    a[1, 2] = torch.nan
    a[4, 6] = torch.nan

    std = a.clone()
    nan_cols = torch.isnan(std).any(dim=0)
    std[torch.isnan(std)] = -torch.inf
    std = std.max(0)[0]
    if propagate_nan == tl.PropagateNan.ALL:
        std[nan_cols] = torch.nan

    ans = torch.zeros((8, ), dtype=torch.float32, device=device)
    max_kernel[1, 1, 1](a, ans, propagate_nan)
    torch.testing.assert_close(std, ans, equal_nan=True)


@pytest.mark.interpreter
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_min_propagate_nan(propagate_nan, device):

    @triton.jit
    def min_kernel(in_ptr, out_ptr, propagate_nan: tl.constexpr):
        a = tl.load(in_ptr + tl.arange(0, 8)[:, None] * 8 + tl.arange(0, 8)[None, :])
        a = tl.min(a, 0, propagate_nan=propagate_nan)
        tl.store(out_ptr + tl.arange(0, 8), a)

    a = torch.randn((8, 8), device=device)
    a[1, 2] = torch.nan
    a[4, 6] = torch.nan

    std = a.clone()
    nan_cols = torch.isnan(std).any(dim=0)
    std[torch.isnan(std)] = torch.inf
    std = std.min(0)[0]
    if propagate_nan == tl.PropagateNan.ALL:
        std[nan_cols] = torch.nan

    ans = torch.zeros((8, ), dtype=torch.float32, device=device)
    min_kernel[1, 1, 1](a, ans, propagate_nan)
    torch.testing.assert_close(std, ans, equal_nan=True)


def _reference_reduce_with_indices(x, op, propagate_nan, tie_break_left):
    values = []
    indices = []
    fill_value = -torch.inf if op == "max" else torch.inf

    for column in x.T:
        nan_indices = torch.nonzero(torch.isnan(column), as_tuple=False).flatten()
        if propagate_nan == tl.PropagateNan.ALL and len(nan_indices) > 0:
            values.append(torch.tensor(torch.nan, device=x.device, dtype=x.dtype))
            indices.append(nan_indices[0 if tie_break_left else -1])
            continue

        clean = column.clone()
        clean[torch.isnan(clean)] = fill_value
        if op == "max":
            value, index = torch.max(clean, dim=0)
        else:
            value, index = torch.min(clean, dim=0)
        values.append(value)
        indices.append(index)

    return torch.stack(values), torch.stack(indices).to(torch.int32)


@pytest.mark.interpreter
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
@pytest.mark.parametrize("tie_break_left", [True, False])
def test_reduce_return_indices_propagate_nan(op, propagate_nan, tie_break_left, device):

    @triton.jit
    def reduce_kernel(in_ptr, out_val_ptr, out_idx_ptr, propagate_nan: tl.constexpr, tie_break_left: tl.constexpr,
                      op: tl.constexpr):
        a = tl.load(in_ptr + tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :])
        if op == "max":
            values, indices = tl.max(a, 0, return_indices=True, return_indices_tie_break_left=tie_break_left,
                                     propagate_nan=propagate_nan)
        else:
            values, indices = tl.min(a, 0, return_indices=True, return_indices_tie_break_left=tie_break_left,
                                     propagate_nan=propagate_nan)
        offs = tl.arange(0, 4)
        tl.store(out_val_ptr + offs, values)
        tl.store(out_idx_ptr + offs, indices)

    x = torch.tensor([[1.0, 5.0, 9.0, 0.0], [torch.nan, -1.0, 3.0, 4.0], [3.0, 2.0, torch.nan, torch.nan],
                      [2.0, torch.nan, 7.0, 6.0]], device=device)

    expected_values, expected_indices = _reference_reduce_with_indices(x, op, propagate_nan, tie_break_left)
    actual_values = torch.empty((4, ), dtype=torch.float32, device=device)
    actual_indices = torch.empty((4, ), dtype=torch.int32, device=device)

    reduce_kernel[(1, )](x, actual_values, actual_indices, propagate_nan, tie_break_left, op)
    torch.testing.assert_close(expected_values, actual_values, equal_nan=True)
    torch.testing.assert_close(expected_indices, actual_indices)


@pytest.mark.interpreter
@pytest.mark.parametrize("op", ["max", "min"])
def test_reduce_return_indices_propagate_nan_tie_break_left(op, device):

    @triton.jit
    def reduce_kernel(in_ptr, out_idx_ptr, op: tl.constexpr):
        a = tl.load(in_ptr + tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :])
        if op == "max":
            _, indices = tl.max(a, 0, return_indices=True, return_indices_tie_break_left=True,
                                propagate_nan=tl.PropagateNan.ALL)
        else:
            _, indices = tl.min(a, 0, return_indices=True, return_indices_tie_break_left=True,
                                propagate_nan=tl.PropagateNan.ALL)
        tl.store(out_idx_ptr + tl.arange(0, 4), indices)

    x = torch.tensor([[1.0, torch.nan, 9.0, 0.0], [torch.nan, -1.0, torch.nan, 4.0], [3.0, torch.nan, 5.0, torch.nan],
                      [2.0, 7.0, torch.nan, 6.0]], device=device)
    expected_indices = torch.tensor([1, 0, 1, 2], dtype=torch.int32, device=device)
    actual_indices = torch.empty((4, ), dtype=torch.int32, device=device)

    reduce_kernel[(1, )](x, actual_indices, op)
    torch.testing.assert_close(expected_indices, actual_indices)
