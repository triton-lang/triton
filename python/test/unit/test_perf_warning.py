import triton
import triton.language as tl
import os
import pytest
import torch
from contextlib import contextmanager


@contextmanager
def enable_remark_context():
    try:
        os.environ['MLIR_ENABLE_REMARK'] = '1'
        yield
    finally:
        os.environ['MLIR_ENABLE_REMARK'] = '0'


def is_perf_warning_enabled():
    return os.environ.get('MLIR_ENABLE_REMARK', '0') == '1'


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def test_mma_remark(capfd, fresh_triton_cache):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            pytest.skip("Requires sm >= 90 to run")

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                        block_shape=(32, 128), order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                        block_shape=(128, 32), order=(0, 1))
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(32, 32), order=(1, 0))
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        c = tl.dot(a, b)
        tl.store(c_block_ptr, c)

    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(
                fn=matmul_kernel, signature={
                    0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9:
                    'i32', 10: 'i32', 11: 'i32'
                }, constants={}))
    captured = capfd.readouterr()

    assert "remark: Warning: can't use MMA V3 for the dot op" in captured.err, "expect MMA V3 remark"
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
    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(fn=ldst_vec, signature={0: '*i64', 1: '*i64', 2: '*fp16', 3: '*fp32', 4: '*fp16'},
                                      constants={"XBLOCK": XBLOCK}), options={"num_warps": 1})

    _, err = capfd.readouterr()

    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"


def test_remark_swp_num_stages_greater_than_loop_iters(capfd, fresh_triton_cache):

    @triton.jit
    def kernel_pipe_num_stages_gt_loop_iters(in_ptr, out_ptr):
        SIZE: tl.constexpr = 64
        in_ptrs = in_ptr + tl.arange(0, SIZE)
        for i in tl.range(0, 2, num_stages=3):
            val = tl.load(in_ptrs)
            in_ptrs += SIZE
            out_ptrs = out_ptr + (tl.arange(0, SIZE) + i * SIZE)
            tl.store(out_ptrs, val)

    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(fn=kernel_pipe_num_stages_gt_loop_iters, signature={0: '*fp32', 1: '*fp32'},
                                      constants={}), options={"cluster_dims": (1, 1, 1)})

    _, err = capfd.readouterr()
    assert ("remark: Warning: fewer loop iterations than pipeline stages. The loop will be treated as a dynamic loop"
            in err), "expect performance warning remark:" + err


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

    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(fn=kernel_pipe_error, signature={0: '*fp32', 1: '*fp32'}, constants={}),
            options={"cluster_dims": (1, 1, 1)})

    _, err = capfd.readouterr()

    import re
    # first match `if tl.max(val) > 0:`, then match `val = tl.load(in_ptrs)``
    pattern = (r"operation scheduled before its operands.*"
               r"note: called from\s+"
               r"if tl.max\(val\) > 0:.*"
               r"The following operation depends on the above operation, which fails the pipelining.\s+"
               r"val = tl.load\(in_ptrs\)")
    # Use re.DOTALL to make '.' match any character including newline
    match = re.search(pattern, err, re.DOTALL)
    # Assert that the pattern is found in the correct order
    assert match, "Expected performance warning remarks not found or not in the correct order:" + err
