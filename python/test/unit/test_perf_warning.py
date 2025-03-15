import os
from contextlib import contextmanager

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda


@contextmanager
def enable_diagnostics_context(value):
    try:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = value
        yield
    finally:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = ""


def test_mma_remark(capfd, fresh_triton_cache):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] != 9:
            pytest.skip("Requires sm = 90 to run")

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    ):
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(32, 128),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, 0),
            block_shape=(128, 32),
            order=(0, 1),
        )
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(0, 0),
            block_shape=(32, 32),
            order=(1, 0),
        )
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        c = tl.dot(a, b)
        tl.store(c_block_ptr, c)

    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    with enable_diagnostics_context('remarks'):
        triton.compile(triton.compiler.ASTSource(
            fn=matmul_kernel,
            signature=signature,
            constexprs={},
        ))
    captured = capfd.readouterr()

    assert ("can't use MMA V3 for the dot op" in captured.err), "expect MMA V3 remark"
    assert "note: see current operation:" not in captured.err

    with enable_diagnostics_context('remarks,operations,stacktraces'):
        triton.compile(triton.compiler.ASTSource(
            fn=matmul_kernel,
            signature=signature,
            constexprs={},
        ))
    captured = capfd.readouterr()
    assert "note: diagnostic emitted with trace:" in captured.err
    assert "note: see current operation:" in captured.err


def test_remark_vectorization(capfd, fresh_triton_cache):

    @triton.jit
    def ldst_vec(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 9
        x2 = (xindex // 3456) % 512
        x1 = (xindex // 9) % 384
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (x2 + (512 * x0)), None, eviction_policy="evict_last")
        tmp1 = tmp0 + 520
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tmp9 = (-4) + tmp3
        tmp12 = tl.full([1], 512, tl.int64)
        tmp14 = tmp9 < tmp12
        tmp16 = tl.load(in_ptr3 + (x1), tmp14, eviction_policy="evict_last", other=0.0)
        tmp18 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp14, tmp19, tmp20)
        tmp22 = tmp21.to(tl.float32)
        tl.store(out_ptr0 + (x4), tmp22, None)

    XBLOCK = 1024

    astsource_args = {
        "fn": ldst_vec,
        "signature": {
            "in_ptr0": "*i64",
            "in_ptr1": "*i64",
            "in_ptr2": "*fp16",
            "in_ptr3": "*fp32",
            "out_ptr0": "*fp16",
            "XBLOCK": "constexpr",
        },
        "constexprs": {"XBLOCK": XBLOCK},
    }

    with enable_diagnostics_context('remarks'):
        triton.compile(
            triton.compiler.ASTSource(**astsource_args),
            options={"num_warps": 1},
        )

    _, err = capfd.readouterr()
    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"
    assert "note: see current operation:" not in err

    with enable_diagnostics_context('remarks,operations,stacktraces'):
        triton.compile(
            triton.compiler.ASTSource(**astsource_args),
            options={"num_warps": 1},
        )

    _, err = capfd.readouterr()
    assert "note: see current operation:" in err
    assert "note: diagnostic emitted with trace:" in err


def test_remark_coalescing(capfd, fresh_triton_cache):

    @triton.jit
    def test_kernel(
        x_ptr,
        x_offsets,
        output_ptr,
        stride_x,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_start = tl.load(x_offsets).to(tl.int64)
        block_ptr = tl.make_block_ptr(
            base=x_ptr + block_start * stride_x,  # block_start * stride_x has divisibility of 1
            shape=(BLOCK_SIZE, ),
            strides=(stride_x, ),
            offsets=(0, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        x = tl.load(block_ptr)  # block_ptr has divisibility of 1

        output_ptr = tl.make_block_ptr(
            base=output_ptr + block_start * stride_x,
            shape=(BLOCK_SIZE, ),
            strides=(stride_x, ),
            offsets=(0, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        tl.store(output_ptr, x)

    size = 4096
    x = torch.rand(size, device=torch.device("cuda")).uniform_(-0.1, 0.1).to(torch.float8_e4m3fn)
    x_offsets = torch.tensor([0], device=torch.device("cuda"))
    output = torch.empty_like(x)
    stride_x = 1
    BLOCK_SIZE = 128
    with enable_diagnostics_context('remarks'):
        test_kernel[(1, )](x, x_offsets, output, stride_x, BLOCK_SIZE=BLOCK_SIZE)

    _, err = capfd.readouterr()
    lines = err.splitlines()
    # Define the expected strings in order
    expected_strings = [
        "one element per thread is assigned", "x = tl.load(block_ptr)",
        "note: The divisibility of the pointer is 1 in all dimensions.", "first introduced here",
        "block_start = tl.load(x_offsets).to(tl.int64)", "add `tt.multiple_of`"
    ]

    # Initialize an index to track the position in expected_strings
    index = 0
    # Iterate over each line in the output
    for line in lines:
        # Check if the current expected string is in the line
        if expected_strings[index] in line:
            # Move to the next expected string
            index += 1
            # If all expected strings have been found, break out of the loop
            if index == len(expected_strings):
                break
    # Check if all expected strings were found
    if index != len(expected_strings):
        missing_string = expected_strings[index]
        raise AssertionError(f"Missing expected string: '{missing_string}' from {err}")


def test_remark_swp_op_before_operands(capfd, fresh_triton_cache):

    @triton.jit
    def kernel_pipe_error(in_ptr, out_ptr):
        SIZE: tl.constexpr = 64
        in_ptrs = in_ptr + tl.arange(0, SIZE)
        val = tl.zeros((SIZE, ), dtype=tl.float32)
        k = 0
        for i in tl.range(0, 64, num_stages=3):
            in_ptrs = in_ptr + tl.arange(0, SIZE) + SIZE * k
            val = tl.load(in_ptrs)
            out_ptrs = out_ptr + (tl.arange(0, SIZE) + i * SIZE)
            tl.store(out_ptrs, val)
            if tl.max(val) > 0:
                k += 1

    i = torch.empty(64 * 64, dtype=torch.float32).cuda()
    o = torch.empty(64 * 64, dtype=torch.float32).cuda()
    kernel_pipe_error[(1, )](i, o)
