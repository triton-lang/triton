import triton
import pytest
import torch
import triton.language as tl

from test_core import _test_binary, int_dtypes, uint_dtypes, float_dtypes, numpy_random

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
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
def test_sort(M, N, descending, dtype_str, device):

    @triton.jit
    def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.sort(x, descending=descending)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.sort(x, descending=descending)[0]
    z = torch.empty_like(x)
    sort_kernel[(1, )](x, z, N, M, descending, num_warps=8)
    assert (y == z).all(), (y, z)


# ---------------
# test flip op
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
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
    x = torch.from_numpy(x).to(device)
    y = torch.flip(x, (1, ))
    z = torch.empty_like(x, device=device)
    flip_kernel[(1, )](x, z, N, M, num_warps=8)
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
