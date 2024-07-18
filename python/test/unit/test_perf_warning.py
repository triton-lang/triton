import triton
import triton.language as tl
import os
import pytest
import torch


def is_perf_warning_enabled():
    return os.environ.get('MLIR_ENABLE_REMARK', '0') == '1'


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def test_remark_mma(capfd):
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


def test_remark_swp_1stage(capfd):
    os.environ['MLIR_ENABLE_REMARK'] = '1'

    @triton.jit
    def vecadd_kernel(a_ptr, b_ptr, output_ptr, n_elements, num_blocks):
        pid = tl.program_id(axis=0)
        block_start = pid * 128 * num_blocks
        offsets = block_start + tl.arange(0, 128)
        for _ in tl.range(0, num_blocks, num_stages=1):
            mask = offsets < n_elements
            x = tl.load(a_ptr + offsets, mask=mask)
            y = tl.load(b_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
            offsets += 128

    triton.compile(
        triton.compiler.ASTSource(
            fn=vecadd_kernel, signature={
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'},
            constants={}))
    _, err = capfd.readouterr()

    assert "remark: Warning: SWP fails. There is no loop with num_stages greater than 1" in err, "expect SWP failure remark"
    # assert "note: see current operation:" in captured.err
    # assert "numstages in loop" in captured.err
    os.environ['MLIR_ENABLE_REMARK'] = '0'


def test_remark_swp_dep_distance(capfd):
    os.environ['MLIR_ENABLE_REMARK'] = '1'

    @triton.jit
    def vecadd_kernel(a_ptr, b_ptr, output_ptr, n_elements, num_blocks):
        pid = tl.program_id(axis=0)
        block_start = pid * 128 * num_blocks
        offsets = block_start + tl.arange(0, 128)
        offsets_0 = block_start + tl.arange(2, 130)
        for _ in tl.range(0, num_blocks):
            mask = offsets < n_elements
            x = tl.load(a_ptr + offsets, mask=mask)
            x_0 = tl.load(a_ptr + offsets_0, mask=mask)
            output = x + x_0
            tl.store(output_ptr + offsets, output, mask=mask)
            offsets += 128
            offsets_0 += 128

    triton.compile(
        triton.compiler.ASTSource(
            fn=vecadd_kernel, signature={
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'},
            constants={}))
    stdout, stderr = capfd.readouterr()

    # TODO: to fix this kernel
    # assert "remark: Warning: SWP fails due to loop distance is greater than" in stdout, "expect SWP failure remark"
    # assert "note: see current operation:" in captured.err
    # assert "numstages in loop" in captured.err
    os.environ['MLIR_ENABLE_REMARK'] = '0'


def test_remark_swp_outerloop(capfd):
    os.environ['MLIR_ENABLE_REMARK'] = '1'

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
            fn=binary_elemwise_kernel_2d, signature={
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, constants={}))
    _, err = capfd.readouterr()

    assert "remark: Warning: SWP fails on the outer loop" in err, "expect SWP failure remark"
    os.environ['MLIR_ENABLE_REMARK'] = '0'
