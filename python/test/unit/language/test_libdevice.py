import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef
from triton._internal_testing import is_cuda


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [
        ("j0", "bessel_j0"),
        ("j1", "bessel_j1"),
        ("y0", "bessel_y0"),
        ("y1", "bessel_y1"),
        ("cyl_bessel_i0", "i0"),
        ("cyl_bessel_i1", "i1"),
    ],
)
def test_bessel(dtype_str, libdevice_fn, torch_special_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    torch.manual_seed(42)
    x = torch.randn((SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    kernel[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)


def test_clz(device):

    @triton.jit
    def triton_clz(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.clz(tl.load(in_ptr + offsets)))

    x = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_clz[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([31, 30, 30, 29, 29, 28, 28, 27], dtype=torch.int32))


def test_popc(device):

    @triton.jit
    def triton_popc(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.popc(tl.load(in_ptr + offsets)))

    x = torch.tensor([0, 1, 2, 3, 4, 7, 8, 15], dtype=torch.int32, device=device)
    y = torch.empty_like(x)

    triton_popc[(1, )](x, y, SIZE=x.numel())

    assert torch.equal(y.cpu(), torch.tensor([0, 1, 1, 2, 1, 3, 1, 4], dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ffs(device, dtype):

    @triton.jit
    def triton_ffs(in_ptr, out_ptr, SIZE: tl.constexpr):
        offsets = tl.arange(0, SIZE)
        tl.store(out_ptr + offsets, libdevice.ffs(tl.load(in_ptr + offsets)))

    bits = torch.iinfo(dtype).bits
    vals = [0, 1, -1, 6, 0x50, torch.iinfo(dtype).min, torch.iinfo(dtype).max]
    vals += [1 << i for i in range(bits - 1)]
    vals += [0] * (triton.next_power_of_2(len(vals)) - len(vals))
    x = torch.tensor(vals, dtype=dtype, device=device)
    y = torch.empty(x.shape, dtype=torch.int32, device=device)

    triton_ffs[(1, )](x, y, SIZE=x.numel())

    # ffs returns the 1-based index of the least significant set bit, or 0.
    ref = [0 if v == 0 else ((v & -v).bit_length()) for v in vals]
    assert torch.equal(y.cpu(), torch.tensor(ref, dtype=torch.int32))


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_isinf(device, dtype_str):

    @triton.jit
    def triton_isinf(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        in_tile = tl.load(in_ptr + offsets, mask=mask)
        if in_ptr.dtype.element_ty == tl.float32:
            out_tile = libdevice.finitef(in_tile)
        else:
            out_tile = libdevice.isfinited(in_tile)
        tl.store(out_ptr + offsets, out_tile, mask=mask)

    x = torch.tensor(
        [float(1), -float(1),
         float(0), -float(0),
         float("inf"), -float("inf"),
         float("nan"), -float("nan")], device=device, dtype=getattr(torch, dtype_str))
    res = torch.tensor([True, True, True, True, False, False, False, False])
    numel = x.numel()
    y = torch.empty_like(x, dtype=torch.bool)
    BLOCK_SIZE = 256
    triton_isinf[(triton.cdiv(numel, BLOCK_SIZE), )](x, y, numel, BLOCK_SIZE)
    assert torch.equal(y.cpu(), res)


@pytest.mark.skipif(not is_cuda(), reason="NVIDIA libdevice functions")
@pytest.mark.parametrize(
    "fn, x, ret_dtype, expected",
    [
        ("float2uint_rn", 3e9, tl.uint32, 3_000_000_000),
        ("float2uint_rz", 3e9, tl.uint32, 3_000_000_000),
        ("double2uint_rd", 3e9, tl.uint32, 3_000_000_000),
        ("double2uint_ru", 3e9, tl.uint32, 3_000_000_000),
        ("float2ull_rn", 2.0**63, tl.uint64, 2**63),
        ("double2ull_rz", 2.0**63, tl.uint64, 2**63),
        ("float_as_uint", -1.0, tl.uint32, 0xBF800000),
    ],
)
def test_libdevice_unsigned_conversions(fn, x, ret_dtype, expected, device):

    @triton.jit
    def kernel(X, Y, FN: tl.constexpr, RET_DTYPE: tl.constexpr):
        y = getattr(libdevice, FN)(tl.load(X))
        tl.static_assert(y.dtype == RET_DTYPE)
        tl.store(Y, y.to(tl.int64))

    x_dtype = torch.float64 if fn.startswith("double") else torch.float32
    x = torch.tensor([x], device=device, dtype=x_dtype)
    y = torch.empty((1, ), device=device, dtype=torch.int64)
    kernel[(1, )](x, y, fn, ret_dtype)
    assert y.item() % 2**64 == expected
