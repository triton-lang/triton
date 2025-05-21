import triton
import triton.language as tl

from triton._internal_testing import (
    numpy_random,
    to_triton,
)
from numpy.random import RandomState


def test_manual_groups(device):

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        g1 = tl.group('g1', 0, 4)
        g2 = tl.group('g2', 4, 4)

        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        with g1:
            x = tl.load(x_ptr + offsets, mask=mask)
        y = x * x
        with g2:
            tl.store(y_ptr + offsets, y, mask=mask)

    shape = 128
    dtype_str = 'float32'
    rs = RandomState(17)
    x = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)
    y = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)

    compiled_kernel = kernel.warmup(x, y, shape, BLOCK_SIZE=1024, grid=(1,))
    ttir = compiled_kernel.asm['ttir']
    assert '"ttg.manual-nvws" = true' in ttir
    assert 'nvws.g1 = {num_warps = 4 : i32, start_warp = 0 : i32}' in ttir
    assert 'nvws.g2 = {num_warps = 4 : i32, start_warp = 4 : i32}' in ttir

    # load has group g1
    assert '%9 = tt.load %8, %6 {groups = [@nvws.g1]}' in ttir

    # store has group g2
    assert 'tt.store %12, %10, %6 {groups = [@nvws.g2]}' in ttir

    # mul has no group
    assert '%10 = arith.mulf %9, %9 : tensor<1024xf32>' in ttir


@triton.jit
def my_func(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, g1: tl.constexpr, g2: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    with g1:
        x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    with g2:
        tl.store(y_ptr + offsets, y, mask=mask)


def test_manual_groups_func_arg(device):

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        g1 = tl.group('g1', 0, 4)
        g2 = tl.group('g2', 4, 4)

        my_func(x_ptr, y_ptr, n_elements, BLOCK_SIZE, g1, g2)

    shape = 128
    dtype_str = 'float32'
    rs = RandomState(17)
    x = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)
    y = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)

    compiled_kernel = kernel.warmup(x, y, shape, BLOCK_SIZE=1024, grid=(1,))
    ttir = compiled_kernel.asm['ttir']
    assert '"ttg.manual-nvws" = true' in ttir
    assert 'nvws.g1 = {num_warps = 4 : i32, start_warp = 0 : i32}' in ttir
    assert 'nvws.g2 = {num_warps = 4 : i32, start_warp = 4 : i32}' in ttir

    # load has group g1
    assert '%9 = tt.load %8, %6 {groups = [@nvws.g1]}' in ttir

    # store has group g2
    assert 'tt.store %12, %10, %6 {groups = [@nvws.g2]}' in ttir

    # mul has no group
    assert '%10 = arith.mulf %9, %9 : tensor<1024xf32>' in ttir

def test_reg_count(device):

    @triton.jit
    def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        g = tl.group('g', 0, 4, 72)

        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        with g:
            x = tl.load(x_ptr + offsets, mask=mask)
        y = x * x
        tl.store(y_ptr + offsets, y, mask=mask)

    shape = 128
    dtype_str = 'float32'
    rs = RandomState(17)
    x = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)
    y = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device, dst_type=dtype_str)

    compiled_kernel = kernel.warmup(x, y, shape, BLOCK_SIZE=1024, grid=(1,))
    ttir = compiled_kernel.asm['ttir']
    assert '"ttg.manual-nvws" = true' in ttir
    assert 'nvws.g = {num_warps = 4 : i32, reg_count = 72 : i32, start_warp = 0 : i32}' in ttir

    # load has group g1
    assert '%9 = tt.load %8, %6 {groups = [@nvws.g]}' in ttir
