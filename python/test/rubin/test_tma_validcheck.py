import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.rubin import mbarrier, tma, fence_async_shared


# In the following tests, the shared buffer is initialized with invalid data.
# To make the tests interesting, we want to ensure that the TMA warp reads
# the invalid data at least once, before the buffer is filled with valid data.
# A simple, atomic-based handshake protocol is used for that purpose. We found that
# this is more robust than having the kernel running and waiting for the valid data in
# a separate stream while the buffer is filled with valid data in the default stream.
@gluon.jit
def request_data_update(flag_ptr):
    ttgl.atomic_cas(flag_ptr, 0, 1)


@gluon.jit
def wait_for_data_update_request(flag_ptr):
    while ttgl.atomic_cas(flag_ptr, 1, 1) != 1:
        pass


@pytest.mark.parametrize("report_validity", ["per_16B_fp16", "per_16B_fp8", "per_elem_1B", "per_16B_fp4"])
def test_tma_check_valid(report_validity):

    @gluon.jit
    def tma_warp(input_ptr, need_update_ptr, smem, bar, XBLOCK: ttgl.constexpr, REPORT_VALIDITY: ttgl.constexpr):
        input_desc = tma.make_tensor_descriptor(
            input_ptr,
            shape=[XBLOCK, XBLOCK],
            strides=[XBLOCK, 1],
            block_shape=[XBLOCK, XBLOCK],
            layout=smem_layout,
        )

        mbarrier.expect(bar, input_desc.block_type.nbytes)
        tma.async_load(input_desc, [0, 0], bar, smem, report_validity=REPORT_VALIDITY)

        primary_phase = 0
        done = 0
        valid = 0

        # Retry after each completed attempt that reports invalid data.
        while done != 1 or valid != 1:
            done, valid = mbarrier.test_wait_validity(bar, primary_phase)

            if done == 1 and valid == 0:
                # After the first call to request_data_update, the input buffer is filled with valid data.
                # This warp keeps retrying TMA until valid data is loaded.
                request_data_update(need_update_ptr)
                primary_phase ^= 1
                mbarrier.expect(bar, input_desc.block_type.nbytes)
                tma.async_load(input_desc, [0, 0], bar, smem, report_validity=REPORT_VALIDITY)

    @gluon.jit
    def consumer(output_ptr, smem, bar, XBLOCK: ttgl.constexpr):
        block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
        xindex = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, block_layout))[:, None]
        yindex = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, block_layout))[None, :]

        # Wait for the conditonal phase to flip - TMA has finished AND the data is valid
        mbarrier.wait(bar, 0, conditional=True)
        val = smem.load(block_layout)
        ttgl.store(output_ptr + yindex + xindex * XBLOCK, val)

    @gluon.jit
    def data_init_warp(input_ptr, valid_input_ptr, need_update_ptr, XBLOCK: ttgl.constexpr):
        # This is a contrived warp solely for a testing purpose. It ensures that
        # the shared buffer is filled with valid data only after the TMA warp loads
        # invalid data.
        copy_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [1], [0])
        copy_width: ttgl.constexpr = 32
        numel: ttgl.constexpr = XBLOCK * XBLOCK

        wait_for_data_update_request(need_update_ptr)

        for i in ttgl.static_range(numel // copy_width):
            offsets = i * copy_width + ttgl.arange(0, copy_width, copy_layout)
            vals = ttgl.load(valid_input_ptr + offsets)
            ttgl.store(input_ptr + offsets, vals)

    @gluon.jit
    def kernel(
        input_ptr,
        valid_input_ptr,
        need_update_ptr,
        output_ptr,
        XBLOCK: ttgl.constexpr,
        smem_layout: ttgl.constexpr,
        dtype: ttgl.constexpr,
        REPORT_VALIDITY: ttgl.constexpr,
    ):
        smem = ttgl.allocate_shared_memory(dtype, [XBLOCK, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)

        ttgl.warp_specialize(
            [
                (consumer, (output_ptr, smem, bar, XBLOCK)),
                (tma_warp, (input_ptr, need_update_ptr, smem, bar, XBLOCK, REPORT_VALIDITY)),
                (data_init_warp, (input_ptr, valid_input_ptr, need_update_ptr, XBLOCK)),
            ],
            (1, 1),
            (128, 32),
        )

        mbarrier.invalidate(bar)

    if report_validity == "per_16B_fp16":
        element_bitwidth = 16
        xblock = 16
        input_tensor = -torch.zeros((xblock, xblock), device="cuda", dtype=torch.float16)
        valid_input = torch.ones_like(input_tensor)
        dtype = ttgl.float16
    elif report_validity == "per_16B_fp8":
        element_bitwidth = 8
        xblock = 32
        invalid_fp32 = -torch.zeros((xblock, xblock), device="cuda", dtype=torch.float32)
        input_tensor = invalid_fp32.to(torch.float8_e4m3fn)
        valid_input = torch.ones((xblock, xblock), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        dtype = ttgl.float8e4nv
    elif report_validity == "per_16B_fp4":
        element_bitwidth = 8
        xblock = 32
        from triton.tools.mxfp import MXFP4Tensor

        invalid_fp32 = -torch.zeros((xblock, xblock * 2), dtype=torch.float32)
        input_tensor = MXFP4Tensor(invalid_fp32).to_packed_tensor(1).to("cuda")
        valid_input = MXFP4Tensor(torch.ones((xblock, xblock * 2), dtype=torch.float32)).to_packed_tensor(1).to("cuda")
        dtype = ttgl.uint8
    elif report_validity == "per_elem_1B":
        element_bitwidth = 8
        xblock = 32
        input_tensor = torch.full((xblock, xblock), 255, dtype=torch.uint8, device="cuda")
        valid_input = torch.ones((xblock, xblock), dtype=torch.uint8, device="cuda")
        dtype = ttgl.uint8
    else:
        raise AssertionError(f"unsupported validity mode: {report_validity}")

    output = torch.empty_like(input_tensor)
    smem_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        element_bitwidth=element_bitwidth,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    need_update = torch.zeros((1, ), device="cuda", dtype=torch.int32)
    kernel[(1, )](
        input_tensor,
        valid_input,
        need_update,
        output,
        xblock,
        smem_layout,
        dtype,
        report_validity,
        num_warps=4,
    )

    torch.testing.assert_close(output, valid_input, rtol=0, atol=0)


def test_multi_stage():
    # Pipelining of loads becomes challenging in the presence of TMA retry - now the TMA warp
    # needs to wait for the completion of TMA that it has issued, and retry if necessary.
    # What's demonstrated here is the most naive solution, involving per-stage state management
    # in the TMA warp. Improving on this solution is left for future work.

    # Helpers for loading and storing per-stage scalar quantities. Since Gluon does not support
    # local array, we use SMEM for the storage.
    @gluon.jit
    def store_scalar(smem, value):
        scalar_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [1], [0])
        zero = ttgl.zeros([1], ttgl.int32, scalar_layout)
        smem.store(ttgl.full_like(zero, value))

    @gluon.jit
    def load_scalar(smem):
        scalar_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [1], [0])
        return smem.load(scalar_layout).item()

    @gluon.jit
    def store_stage_state(stage_state, field: ttgl.constexpr, stage, num_stages: ttgl.constexpr, value):
        slot: ttgl.constexpr = field * num_stages
        store_scalar(stage_state.index(slot + stage), value)

    @gluon.jit
    def load_stage_state(stage_state, field: ttgl.constexpr, stage, num_stages: ttgl.constexpr):
        slot: ttgl.constexpr = field * num_stages
        return load_scalar(stage_state.index(slot + stage))

    @gluon.jit
    def tma_warp(
        input_ptr,
        need_update_ptr,
        num_elems: ttgl.constexpr,
        block_size: ttgl.constexpr,
        num_stages: ttgl.constexpr,
        smem,
        bar_full,
        bar_empty,
        REPORT_VALIDITY: ttgl.constexpr,
    ):
        input_desc = tma.make_tensor_descriptor(
            input_ptr,
            shape=[1, num_elems],
            strides=[num_elems, 1],
            block_shape=[1, block_size],
            layout=smem_layout,
        )

        PRIMARY_PHASE = ttgl.constexpr(0)
        DONE = ttgl.constexpr(1)
        NUM_STAGE_STATE_FIELDS = ttgl.constexpr(2)

        # Keep track of state[num_stages][num_states] as an 1D "array"
        stage_state = ttgl.allocate_shared_memory(
            ttgl.int32,
            [NUM_STAGE_STATE_FIELDS * num_stages, 1],
            mbarrier.MBarrierLayout(),
        )

        for i in range(num_stages):
            store_stage_state(stage_state, PRIMARY_PHASE, i, num_stages, 0)

        offset = 0
        num_iter = num_elems // block_size
        acquire_phase = 0

        for _ in range(num_iter // num_stages):
            acquire_phase ^= 1

            # Acquire and initially fill every stage in this batch.
            for stage in ttgl.static_range(num_stages):
                cur_full_bar = bar_full.index(stage)
                mbarrier.wait(bar_empty.index(stage), acquire_phase)
                mbarrier.expect(cur_full_bar, input_desc.block_type.nbytes)
                tma.async_load(
                    input_desc,
                    [0, offset + block_size * stage],
                    cur_full_bar,
                    smem.slice(stage, 1, dim=0),
                    report_validity=REPORT_VALIDITY,
                )
                store_stage_state(stage_state, DONE, stage, num_stages, 0)

            num_done = 0
            while num_done != num_stages:
                for stage in ttgl.static_range(num_stages):
                    # Has this stage observed valid data in this batch?
                    if load_stage_state(stage_state, DONE, stage, num_stages) == 0:
                        cur_full_bar = bar_full.index(stage)
                        primary_phase = load_stage_state(stage_state, PRIMARY_PHASE, stage, num_stages)
                        attempt_done, data_valid = mbarrier.test_wait_validity(cur_full_bar, primary_phase)

                        if attempt_done == 1:
                            # TMA has completed for this primary phase and the stage
                            store_stage_state(stage_state, PRIMARY_PHASE, stage, num_stages, primary_phase ^ 1)

                            if data_valid:
                                num_done += 1
                                store_stage_state(stage_state, DONE, stage, num_stages, 1)
                            else:
                                # The primary phase has advanced but need a retry, check validity after the next
                                # primary phase has completed
                                request_data_update(need_update_ptr)
                                mbarrier.expect(cur_full_bar, input_desc.block_type.nbytes)
                                tma.async_load(
                                    input_desc,
                                    [0, offset + block_size * stage],
                                    cur_full_bar,
                                    smem.slice(stage, 1, dim=0),
                                    report_validity=REPORT_VALIDITY,
                                )

            offset += num_stages * block_size

    @gluon.jit
    def data_init_warp(
        input_ptr,
        valid_input_ptr,
        order_ptr,
        need_update_ptr,
        num_elems: ttgl.constexpr,
        block_size: ttgl.constexpr,
    ):
        copy_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [1], [0])
        copy_width: ttgl.constexpr = 32
        num_blocks: ttgl.constexpr = num_elems // block_size

        wait_for_data_update_request(need_update_ptr)

        for i in ttgl.static_range(num_blocks):
            block_idx = ttgl.load(order_ptr + i).item()
            block_offset = block_idx * block_size
            for j in ttgl.static_range(block_size // copy_width):
                offsets = block_offset + j * copy_width + ttgl.arange(0, copy_width, copy_layout)
                vals = ttgl.load(valid_input_ptr + offsets)
                ttgl.store(input_ptr + offsets, vals)

    @gluon.jit
    def consumer(
        output_ptr,
        num_elems: ttgl.constexpr,
        block_size: ttgl.constexpr,
        num_stages: ttgl.constexpr,
        smem,
        bar_full,
        bar_empty,
    ):
        block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
        offset = ttgl.arange(0, block_size)

        num_iter = num_elems // block_size
        phase = 0

        for _ in range(num_iter // num_stages):
            for stage in ttgl.static_range(num_stages):
                mbarrier.wait(bar_full.index(stage), phase, conditional=True)

                value = smem.slice(stage, 1, dim=0).load(block_layout)
                value = ttgl.reshape(value, [block_size])

                fence_async_shared()
                mbarrier.arrive(bar_empty.index(stage), count=1)
                ttgl.store(output_ptr + offset, value)
                offset += block_size

            phase ^= 1

    @gluon.jit
    def kernel(
        input_ptr,
        valid_input_ptr,
        order_ptr,
        need_update_ptr,
        output_ptr,
        num_elems: ttgl.constexpr,
        block_size: ttgl.constexpr,
        num_stages: ttgl.constexpr,
        smem_layout: ttgl.constexpr,
        dtype: ttgl.constexpr,
        REPORT_VALIDITY: ttgl.constexpr,
    ):
        smem = ttgl.allocate_shared_memory(dtype, [num_stages, block_size], smem_layout)
        bar_full = ttgl.allocate_shared_memory(ttgl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
        bar_empty = ttgl.allocate_shared_memory(ttgl.int64, [num_stages, 1], mbarrier.MBarrierLayout())

        for i in range(num_stages):
            mbarrier.init(bar_full.index(i), count=1)
            mbarrier.init(bar_empty.index(i), count=1)

        ttgl.warp_specialize(
            [
                (consumer, (output_ptr, num_elems, block_size, num_stages, smem, bar_full, bar_empty)),
                (tma_warp, (input_ptr, need_update_ptr, num_elems, block_size, num_stages, smem, bar_full, bar_empty,
                            REPORT_VALIDITY)),
                (data_init_warp, (input_ptr, valid_input_ptr, order_ptr, need_update_ptr, num_elems, block_size)),
            ],
            (1, 1),
            (128, 32),
        )

    report_validity = "per_16B_fp16"
    element_bitwidth = 16
    num_elems = 1024
    block_size = 128
    num_stages = 4

    input_tensor = -torch.zeros((1, num_elems), device="cuda", dtype=torch.float16)
    valid_input = torch.empty_like(input_tensor)
    num_blocks = num_elems // block_size
    for block in range(num_blocks):
        valid_input[0, block * block_size:(block + 1) * block_size] = block

    output = torch.empty_like(input_tensor)
    smem_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=0,
        element_bitwidth=element_bitwidth,
        rank=2,
        transposed=False,
        fp4_padded=False,
    )

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    need_update = torch.zeros((1, ), device="cuda", dtype=torch.int32)
    order = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)

    kernel[(1, )](
        input_tensor,
        valid_input,
        order,
        need_update,
        output,
        num_elems,
        block_size,
        num_stages,
        smem_layout,
        ttgl.float16,
        report_validity,
        num_warps=4,
    )
    torch.testing.assert_close(output, valid_input, rtol=0, atol=0)
