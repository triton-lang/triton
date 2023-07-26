import pytest

import torch

import triton
import triton.language as tl


######
# Legacy pointer with explicitly specified pointer info
######
def test_legacy_pointer_explicit():
    @triton.jit
    def legacy_pointer_explicit(
        x_ptr,
        y_ptr,
        PTR_INFO: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        x_ptr = tl.set_ptr_info(x_ptr, PTR_INFO)
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y_ptr = tl.set_ptr_info(y_ptr, PTR_INFO)
        tl.store(y_ptr + offsets, x)

    shape = (128, )
    ptr_info = { "test_key" : "test_value" }
    x = torch.zeros(shape, dtype=torch.int32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    run = legacy_pointer_explicit[(1,)]
    print(run)
    run(x, y, BLOCK_SIZE=shape[0], PTR_INFO=ptr_info)


######
# Block pointer with explicitly specified pointer info
######
def test_block_pointer_explicit():
    @triton.jit
    def block_pointer_explicit(
        x_ptr,
        y_ptr,
        PTR_INFO: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        x_ptr = tl.set_ptr_info(x_ptr, PTR_INFO)
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(BLOCK_SIZE),
            strides=(1),
            offsets=(0),
            block_shape=(BLOCK_SIZE),
            order=(0),
        )
        x = tl.load(x_block_ptr, boundary_check=(0))
        # intentionally leave y_ptr without ptr_info
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(BLOCK_SIZE),
            strides=(1),
            offsets=(0),
            block_shape=(BLOCK_SIZE),
            order=(0),
        )
        tl.store(y_block_ptr, x, boundary_check=(0))

    shape = (128, )
    ptr_info = { "test_key" : "test_value" }
    x = torch.zeros(shape, dtype=torch.float32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    block_pointer_explicit[(1,)](x, y, BLOCK_SIZE=shape[0], PTR_INFO=ptr_info)
