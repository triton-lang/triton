import triton
import triton.language as tl
import os
import pytest
import torch


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
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9:
                'i32', 10: 'i32', 11: 'i32'
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
        triton.compiler.ASTSource(fn=ldst_vec, signature={0: '*i64', 1: '*i64', 2: '*fp16', 3: '*fp32', 4: '*fp16'},
                                  constants={"XBLOCK": XBLOCK}), options={"num_warps": 1})

    _, err = capfd.readouterr()
    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"
    os.environ["MLIR_ENABLE_REMARK"] = "0"


def test_remark_swp_outerloop(capfd):
    os.environ["MLIR_ENABLE_REMARK"] = "1"

    @triton.jit
    def binary_elemwise_kernel_2d(x_ptr, y_ptr, o_ptr, num_elem, shape_x0, shape_x1, stride):
        pid = tl.program_id(axis=0)
        start_elem = min(pid * num_elem, shape_x0)
        end_elem = min(pid * num_elem + num_elem, shape_x0)

        # The outer-loop will not be pipelined in the precondition check,
        # There will be warning for the outer loop.
        for i in range(start_elem, end_elem):
            x_addr = x_ptr + i * stride
            y_addr = y_ptr + i * stride
            o_addr = o_ptr + i * stride

            elem_offset = tl.arange(0, 128)
            x_blk_ptr = x_addr + elem_offset
            y_blk_ptr = y_addr + elem_offset
            o_blk_ptr = o_addr + elem_offset

            block_start = 0
            for _ in range(0, shape_x1, 128):
                elem_offset = block_start + tl.arange(0, 128)
                mask = elem_offset < shape_x1

                x = tl.load(x_blk_ptr, mask=mask)
                y = tl.load(y_blk_ptr, mask=mask)

                output = x + y

                tl.store(o_blk_ptr, output, mask=mask)

                x_blk_ptr += 128
                y_blk_ptr += 128
                o_blk_ptr += 128
                block_start += 128

    triton.compile(
        triton.compiler.ASTSource(
            fn=binary_elemwise_kernel_2d,
            signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, constants={}))

    _, err = capfd.readouterr()

    assert ("remark: Warning: SWP fails on the outer loop" in err), "expect SWP failure remark"
    os.environ["MLIR_ENABLE_REMARK"] = "0"
