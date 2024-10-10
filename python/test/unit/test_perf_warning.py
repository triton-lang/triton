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


def test_mma_remark(capfd):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            pytest.skip("Requires sm >= 90 to run")

    os.environ['MLIR_ENABLE_REMARK'] = '1'

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

    triton.compile(
        triton.compiler.ASTSource(
            fn=matmul_kernel, signature={
                'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32', 'M': 'i32', 'N': 'i32', 'K': 'i32', 'stride_am':
                'i32', 'stride_ak': 'i32', 'stride_bk': 'i32', 'stride_bn': 'i32', 'stride_cm': 'i32', 'stride_cn':
                'i32'
            }, constants={}))
    captured = capfd.readouterr()

    assert "remark: Warning: can't use MMA V3 for the dot op" in captured.err, "expect MMA V3 remark"
    assert "note: see current operation:" in captured.err
    os.environ['MLIR_ENABLE_REMARK'] = '0'


def test_remark_vectorization(capfd):
    os.environ["MLIR_ENABLE_REMARK"] = "1"

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
    triton.compile(
        triton.compiler.ASTSource(
            fn=ldst_vec, signature={
                'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16'
            }, constants={"XBLOCK": XBLOCK}), options={"num_warps": 1})

    _, err = capfd.readouterr()
    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"
    os.environ["MLIR_ENABLE_REMARK"] = "0"


def test_remark_size_per_thread_equals_one(capfd, fresh_triton_cache):

    @triton.jit
    def triton_per_fused_sum(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
        xnumel: tl.constexpr = 8134407
        rnumel: tl.constexpr = 33
        RBLOCK: tl.constexpr = 64
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        rindex = tl.arange(0, RBLOCK)[None, :]
        rmask = rindex < rnumel
        tmp0 = tl.load(in_ptr0 + (rindex + (rnumel * xindex)), rmask & xmask, other=0.0)
        tmp4 = tl.sum(tmp0, 1)[:, None]
        tl.store(out_ptr0 + (xindex), tmp4, xmask)

    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(fn=triton_per_fused_sum, signature={'in_ptr0': '*fp32', 'out_ptr0': '*fp32'},
                                      constants={'XBLOCK': 128}), options={
                                          "cluster_dims": (
                                              63551,  # tl.cdiv(xnumel, XBLOCK)
                                              1, 1)
                                      })

    _, err = capfd.readouterr()
    assert ("remark: Warning: loading only 1 element per thread."
            in err), "expect performance warning remark:" + err
    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"


def test_remark_size_per_thread_equals_one_(capfd, fresh_triton_cache):

    @triton.jit
    def triton_per_fused_sum(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
        xnumel: tl.constexpr = 8134408
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, xmask, other=0.0)
        tmp4 = tl.sum(tmp0, 0)
        tl.store(out_ptr0 + xindex, tmp4, xmask)

    with enable_remark_context():
        triton.compile(
            triton.compiler.ASTSource(fn=triton_per_fused_sum, signature={'in_ptr0': '*fp32', 'out_ptr0': '*fp32'},
                                      constants={'XBLOCK': 128}), options={
                                          "cluster_dims": (
                                              63551,  # tl.cdiv(xnumel, XBLOCK)
                                              1, 1)
                                      })

    _, err = capfd.readouterr()
    assert ("remark: Warning: loading only 1 element per thread."
            in err), "expect performance warning remark:" + err
