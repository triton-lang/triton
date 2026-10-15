# ruff: noqa: E402
import sys
import os

os.environ["TRITON_INSTRUMENTATION_MODE"] = "consan"
import torch
from triton import knobs
from triton.runtime._allocation import set_profile_allocator
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

knobs.compilation.instrumentation_mode = "consan"
knobs.refresh_knobs()


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


set_profile_allocator(alloc_fn)


def barrier_storage_lifetime():
    mode = sys.argv[2]

    @gluon.jit
    def kernel(output_ptr, MODE: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        barrier_layout: ttgl.constexpr = ttgl.amd.cdna5.mbarrier.MBarrierLayout()
        storage = ttgl.allocate_shared_memory(ttgl.int32, [2], shared_layout)
        barrier = storage.reinterpret(ttgl.int64, [1], barrier_layout)
        ttgl.amd.cdna5.mbarrier.init(barrier, count=128)

        if MODE == "read":
            values = storage.load(layout)
            ttgl.store(output_ptr + ttgl.arange(0, 2, layout), values)
        elif MODE == "atomic":
            indices = ttgl.full([2], 0, ttgl.int32, layout)
            values = ttgl.full([2], 1, ttgl.int32, layout)
            storage.atomic_scatter_add(values, indices, axis=0)
        elif MODE == "reinitialize":
            ttgl.amd.cdna5.mbarrier.init(barrier, count=128)
            ttgl.amd.cdna5.mbarrier.arrive(barrier, count=1)
            ttgl.amd.cdna5.mbarrier.wait(barrier, phase=0)
        else:
            values = ttgl.full([2], 7, ttgl.int32, layout)
            if MODE == "partial":
                storage.slice(1, 1).store(ttgl.full([1], 7, ttgl.int32, layout))
            else:
                storage.store(values)
            if MODE == "use-after-store" or MODE == "partial":
                ttgl.amd.cdna5.mbarrier.arrive(barrier, count=1)
            else:
                values = storage.load(layout)
                ttgl.store(output_ptr + ttgl.arange(0, 2, layout), values)

    output = torch.empty((2, ), dtype=torch.int32, device="cuda")
    kernel[(1, )](output, MODE=mode, num_warps=4)
    if mode == "store":
        torch.testing.assert_close(output, torch.full_like(output, 7))


def barrier_storage_pending_completion():
    wait = sys.argv[2] == "True"

    @gluon.jit
    def kernel(input_ptr, WAIT: ttgl.constexpr):
        num_warps: ttgl.constexpr = 4
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        barrier_layout: ttgl.constexpr = ttgl.amd.cdna5.mbarrier.MBarrierLayout()
        desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(
            base=input_ptr,
            shape=(block, block),
            strides=(block, 1),
            block_shape=(block, block),
            layout=shared_layout,
        )
        payload = ttgl.allocate_shared_memory(ttgl.float16, [block, block], shared_layout)
        storage = ttgl.allocate_shared_memory(ttgl.int32, [2], barrier_layout)
        barrier = storage.reinterpret(ttgl.int64, [1], barrier_layout)
        ttgl.amd.cdna5.mbarrier.init(barrier, count=num_warps)
        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], payload, mbarrier=barrier)
        if WAIT:
            ttgl.amd.cdna5.mbarrier.wait(barrier, phase=0)
        storage.store(ttgl.full([2], 7, ttgl.int32, layout))

    data = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    kernel[(1, )](data, WAIT=wait, num_warps=4)


def deadlock_two_partitions():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def deadlock_overarrival():

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

    kernel[(1, )](num_warps=4)


def deadlock_underarrival():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=8 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=8 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def deadlock_different_phases():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=1)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def barrier_underflow():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=3)
        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


XBLOCK = 128


def aliasing_shared_visibility():
    MISSING_BAR = sys.argv[2] == "True"
    OVERLAP = sys.argv[3] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def writer(alias0: ttgl.constexpr, bar: ttgl.constexpr, OVERLAP: ttgl.constexpr, blocked_layout: ttgl.constexpr):
        SIZE_N: ttgl.constexpr = XBLOCK_C * 2 if OVERLAP else XBLOCK_C
        vals = ttgl.full([XBLOCK_C, SIZE_N], 42.0, ttgl.float16, blocked_layout)
        alias0.store(vals)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def reader(alias1: ttgl.constexpr, dummy: ttgl.constexpr, bar: ttgl.constexpr, MISSING_BAR: ttgl.constexpr,
               blocked_layout: ttgl.constexpr):
        if not MISSING_BAR:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        val = alias1.load(blocked_layout)
        dummy.store(val)

    @gluon.jit
    def kernel(MISSING_BAR: ttgl.constexpr, OVERLAP: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK_C], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C * 2], smem_layout)
        smem2 = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        alias0 = smem if OVERLAP else smem.slice(0, XBLOCK_C, dim=1)
        alias1 = smem.slice(XBLOCK_C, XBLOCK_C, dim=1)
        ttgl.warp_specialize([
            (writer, (alias0, bar, OVERLAP, blocked_layout)),
            (reader, (alias1, smem2, bar, MISSING_BAR, blocked_layout)),
        ], [4])

    kernel[(1, )](MISSING_BAR=MISSING_BAR, OVERLAP=OVERLAP, num_warps=4)


def ws_two_loads_two_bars():
    MISSING_BAR = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        if MISSING_BAR != "1":
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for i in range(3):
            ttgl.amd.cdna5.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4])
        ttgl.amd.cdna5.mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK_C, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device="cuda", dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


def ws_two_loads_one_bar():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=2 * 4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, blocked_layout)),
            (ws_1, (smem, bar, blocked_layout)),
            (ws_2, (smem, bar, FAILURE, blocked_layout)),
        ], [4, 4])
        ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK_C, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device="cuda", dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


def ws_two_loads_two_bars_loop():
    MISSING_BAR = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK_C], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "2":
                ttgl.amd.cdna5.mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
            acc = acc + val
        smem.index(1).store(acc)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK_C], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "3":
                ttgl.amd.cdna5.mbarrier.wait(bar.index(3), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)
            acc = acc + val
        smem.index(2).store(acc)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "0":
                ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=phase)
            if MISSING_BAR != "1":
                ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(3), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [4, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for i in range(4):
            ttgl.amd.cdna5.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(3), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4])

    output = torch.empty((XBLOCK, ), device="cuda", dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


def ws_load_ordering():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            ttgl.amd.cdna5.mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
            smem.index(1).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK_C], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=phase)
            val = smem.index(1 if FAILURE else 0).load(layout)
            ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)
            acc = acc + val
        smem.index(2).store(acc)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for i in range(3):
            ttgl.amd.cdna5.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4])

    output = torch.empty((XBLOCK, ), device="cuda", dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


def ws_different_warp_sizes():
    MISSING_BAR = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        if MISSING_BAR != "1":
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        layout_4: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4],
                                                      order=[0])
        layout_2: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[2],
                                                      order=[0])
        layout_8: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[8],
                                                      order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=2 * WARP_SIZE)
        ttgl.amd.cdna5.mbarrier.init(bar.index(2), count=8 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, layout_4)),
            (ws_1, (smem, bar, MISSING_BAR, layout_2)),
            (ws_2, (smem, bar, MISSING_BAR, layout_8)),
        ], [2, 8])
        ttgl.amd.cdna5.mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(layout_4)
        output_ptrs = output + ttgl.arange(0, XBLOCK_C, layout_4)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device="cuda", dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


def async_tdm_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, out, FAILURE: ttgl.constexpr):
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                         strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                         layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=NUM_WARPS)

        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem, mbarrier=bar.index(0))
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, val)

    input = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16)
    input = input.cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16)
    output = output.cuda()
    kernel[(1, )](input, output, FAILURE=FAILURE, num_warps=4)


def async_tdm_kernel_2bufs_1bar():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(a_ptr, b_ptr, out, FAILURE: ttgl.constexpr):
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        a_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=a_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        b_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=b_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        a_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=a_desc.block_shape, layout=a_desc.layout)
        b_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=b_desc.block_shape, layout=b_desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        # 2 TDM loads x NUM_WARPS per-warp arrivals = 2*NUM_WARPS total
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=2 * NUM_WARPS)

        ttgl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem, mbarrier=bar.index(0))
        ttgl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem, mbarrier=bar.index(0))
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        val = a_smem.load(blocked_layout)
        val = val + b_smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, val)

    a = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    b = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](a, b, output, FAILURE=FAILURE, num_warps=4)


def tdm_interleave_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, out, FAILURE: ttgl.constexpr):
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                         strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                         layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2] + list(desc.block_shape), desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=NUM_WARPS)
        ttgl.amd.cdna5.mbarrier.init(bar.index(1), count=NUM_WARPS)

        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem.index(0), mbarrier=bar.index(0))
        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem.index(1), mbarrier=bar.index(1))

        ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, smem.index(0).load(blocked_layout))
        ttgl.store(out_ptr, smem.index(1).load(blocked_layout))

        out_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        ttgl.amd.cdna5.tdm.async_store(out_desc, [0, 0], smem.index(0))
        ttgl.amd.cdna5.tdm.async_wait(0)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, FAILURE=FAILURE, num_warps=4)


def async_copy_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, FAILURE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [XBLOCK_C, XBLOCK_C], [1, 0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C, XBLOCK_C], smem_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        offs_m = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
        offs = offs_m * XBLOCK_C + offs_n
        ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), input_ptr + offs)
        ttgl.amd.cdna5.async_copy.commit_group()

        ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(1), input_ptr + offs)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.wait_group(2 if FAILURE else 1)

        ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), input_ptr + offs)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.wait_group(0)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, FAILURE=FAILURE, num_warps=4)


def async_wait_count_bounds():
    mode = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, output_ptr, MODE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [XBLOCK_C, XBLOCK_C], [1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        offs_m = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
        offs = offs_m * XBLOCK_C + offs_n
        if MODE == "empty_groups":
            smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                               ttgl.load(input_ptr + offs))
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, smem)
            ttgl.amd.cdna5.async_copy.commit_group()
            for _ in range(130):
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(0)
            smem.store(ttgl.full([XBLOCK_C, XBLOCK_C], 0, ttgl.float16, blocked_layout))
        elif MODE == "large_count":
            smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.global_to_shared(smem, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(200)
            ttgl.store(output_ptr + offs, smem.load(blocked_layout))
            ttgl.amd.cdna5.async_copy.wait_group(0)
        else:
            tdm_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=tdm_layout)
            smem = ttgl.allocate_shared_memory(ttgl.float16, desc.block_shape, desc.layout)
            bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
            ttgl.amd.cdna5.mbarrier.init(bar, count=num_warps)
            ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem, mbarrier=bar)
            ttgl.amd.cdna5.async_copy.wait_group(-1)
            ttgl.store(output_ptr + offs, smem.load(blocked_layout))
            ttgl.amd.cdna5.mbarrier.wait(bar, phase=0)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, MODE=mode, num_warps=4)
    if mode == "empty_groups":
        torch.testing.assert_close(output, input_t)


def async_wait_uncommitted():
    mode = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, output_ptr, MODE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [XBLOCK_C, XBLOCK_C], [1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        offs_m = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
        offs = offs_m * XBLOCK_C + offs_n
        if MODE == "load":
            smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(1), input_ptr + offs)
            ttgl.amd.cdna5.async_copy.wait_group(-1)
            ttgl.store(output_ptr + offs, smem.index(0).load(blocked_layout))
            ttgl.store(output_ptr + XBLOCK_C * XBLOCK_C + offs, smem.index(1).load(blocked_layout))
        elif MODE == "store":
            smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                               ttgl.load(input_ptr + offs))
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, smem)
            ttgl.amd.cdna5.async_copy.wait_group(-1)
            smem.store(ttgl.full([XBLOCK_C, XBLOCK_C], 0, ttgl.float16, blocked_layout))
        elif MODE == "store_wait_zero":
            smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                               ttgl.load(input_ptr + offs))
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, smem)
            ttgl.amd.cdna5.async_copy.wait_group(0)
            smem.store(ttgl.full([XBLOCK_C, XBLOCK_C], 0, ttgl.float16, blocked_layout))
            ttgl.amd.cdna5.async_copy.wait_group(-1)
        else:
            smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.global_to_shared(smem, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.wait_group(0)
            ttgl.store(output_ptr + offs, smem.load(blocked_layout))
            ttgl.amd.cdna5.async_copy.wait_group(-1)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((2, XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, MODE=mode, num_warps=4)
    if mode == "load":
        torch.testing.assert_close(output, input_t.expand(2, XBLOCK, XBLOCK))
    elif mode == "store":
        torch.testing.assert_close(output[0], input_t)


def async_copy_load_store():
    mode = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, output_ptr, MODE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [XBLOCK_C, XBLOCK_C], [1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        offs_m = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
        offs = offs_m * XBLOCK_C + offs_n
        zeros = ttgl.full([XBLOCK_C, XBLOCK_C], 0, ttgl.float16, blocked_layout)
        source = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                             ttgl.load(input_ptr + offs))
        if MODE == "interleaved":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(0)
            ttgl.store(output_ptr + XBLOCK_C * XBLOCK_C + offs, loaded.load(blocked_layout))
            source.store(zeros)
        elif MODE == "read_store_source":
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + XBLOCK_C * XBLOCK_C + offs, source)
            ttgl.store(output_ptr + 2 * XBLOCK_C * XBLOCK_C + offs, source.load(blocked_layout))
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(0)
            source.store(zeros)
        elif MODE == "load_into_store_source":
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(source, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(0)
        elif MODE == "mixed_partial_wait_store":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(1)
            source.store(zeros)
            ttgl.amd.cdna5.async_copy.wait_group(0)
        elif MODE == "mixed_partial_wait_load":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(1)
            ttgl.store(output_ptr + XBLOCK_C * XBLOCK_C + offs, loaded.load(blocked_layout))
            ttgl.amd.cdna5.async_copy.wait_group(0)
        elif MODE == "mixed_uncommitted_load":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.wait_group(0)
            source.store(zeros)
            ttgl.amd.cdna5.async_copy.wait_group(-1)
        elif MODE == "mixed_partial_wait_after_arrive":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout)
            bar = ttgl.allocate_shared_memory(ttgl.int64, [1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
            ttgl.amd.cdna5.mbarrier.init(bar, count=num_warps * 32)
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded, input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bar)
            ttgl.amd.cdna5.mbarrier.wait(bar, phase=0)
            loaded.store(zeros)
            ttgl.amd.cdna5.async_copy.wait_group(1)
            source.store(zeros)
            ttgl.amd.cdna5.async_copy.wait_group(0)
        elif MODE == "mixed_older_store":
            loaded = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C, XBLOCK_C], smem_layout)
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded.index(0), input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.global_to_shared(loaded.index(1), input_ptr + offs)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(1)
            ttgl.store(output_ptr + XBLOCK_C * XBLOCK_C + offs, loaded.index(0).load(blocked_layout))
            ttgl.amd.cdna5.async_copy.wait_group(0)
            source.store(zeros)
        else:
            younger = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                                  ttgl.load(input_ptr + offs))
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + offs, source)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + XBLOCK_C * XBLOCK_C + offs, younger)
            ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(1)
            if MODE == "store_partial_wait":
                source.store(zeros)
            else:
                younger.store(zeros)
            ttgl.amd.cdna5.async_copy.wait_group(0)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((3, XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, MODE=mode, num_warps=4)
    if mode == "read_store_source":
        torch.testing.assert_close(output, input_t.expand(3, XBLOCK, XBLOCK))
    elif mode in ("interleaved", "store_partial_wait", "mixed_older_store"):
        torch.testing.assert_close(output[:2], input_t.expand(2, XBLOCK, XBLOCK))


def async_copy_partition_isolation():
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def get_offsets(layout: ttgl.constexpr):
        offs_m = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=1, parent=layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK_C, layout=ttgl.SliceLayout(dim=0, parent=layout))[None, :]
        return offs_m * XBLOCK_C + offs_n

    @gluon.jit
    def loader(input_ptr, output_ptr, loaded, bar, layout: ttgl.constexpr):
        offs = get_offsets(layout)
        ttgl.amd.cdna5.mbarrier.wait(bar, phase=0)
        ttgl.amd.cdna5.async_copy.global_to_shared(loaded.index(0), input_ptr + offs)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.global_to_shared(loaded.index(1), input_ptr + offs)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.wait_group(1)
        ttgl.store(output_ptr + XBLOCK_C * XBLOCK_C + offs, loaded.index(0).load(layout))
        ttgl.amd.cdna5.async_copy.wait_group(0)

    @gluon.jit
    def storer(output_ptr, source, bar, layout: ttgl.constexpr):
        # The barrier orders this uncommitted store before the loader's wait.
        ttgl.amd.cdna5.async_copy.shared_to_global(output_ptr + get_offsets(layout), source)
        ttgl.amd.cdna5.mbarrier.arrive(bar, count=1)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.wait_group(0)

    @gluon.jit
    def kernel(input_ptr, output_ptr):
        WARP_SIZE: ttgl.constexpr = 32
        num_warps: ttgl.constexpr = ttgl.num_warps()
        smem_layout: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [XBLOCK_C, XBLOCK_C], [1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
        source = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK_C, XBLOCK_C], smem_layout,
                                             ttgl.load(input_ptr + get_offsets(blocked_layout)))
        loaded = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bar.index(0), count=num_warps * WARP_SIZE)
        ttgl.warp_specialize([
            (loader, (input_ptr, output_ptr, loaded, bar.index(0), blocked_layout)),
            (storer, (output_ptr, source, bar.index(0), blocked_layout)),
        ], [num_warps])

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((2, XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, num_warps=4)
    torch.testing.assert_close(output, input_t.expand(2, XBLOCK, XBLOCK))


@gluon.jit
def tdm_load_tile(tile_source, mbarrier=None):
    tile_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=tile_source, shape=(32, 32), strides=(32, 1),
                                                     block_shape=(32, 32), layout=tile_layout)
    tile = ttgl.allocate_shared_memory(ttgl.float16, [32, 32], tile_layout)
    ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], tile, mbarrier=mbarrier)
    return tile


def async_copy_wait_pending_tdm():
    wait_tdm = sys.argv[2] == "True"

    @gluon.jit
    def kernel(source, tile_source, output, tile_output, WAIT_TDM: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        block: ttgl.constexpr = 32 * num_warps
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])
        tile_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])
        offsets = ttgl.arange(0, block, layout)
        bars = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(bars.index(0), count=num_warps)
        ttgl.amd.cdna5.mbarrier.init(bars.index(1), count=block)
        tile = tdm_load_tile(tile_source, mbarrier=bars.index(0))
        copy = ttgl.allocate_shared_memory(ttgl.int32, [block], ttgl.SwizzledSharedLayout(1, 1, 1, [0]))
        ttgl.amd.cdna5.async_copy.global_to_shared(copy, source + offsets)
        ttgl.amd.cdna5.async_copy.commit_group()
        ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(1))
        ttgl.amd.cdna5.async_copy.wait_group(-1)
        if WAIT_TDM:
            ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
        ttgl.store(output + offsets, copy.load(layout))
        # Waiting for async copies does not complete the TDM load.
        values = tile.load(tile_layout)
        rows = ttgl.arange(0, 32, ttgl.SliceLayout(1, tile_layout))
        cols = ttgl.arange(0, 32, ttgl.SliceLayout(0, tile_layout))
        ttgl.store(tile_output + rows[:, None] * 32 + cols[None, :], values)
        ttgl.amd.cdna5.mbarrier.wait(bars.index(1), phase=0)

    source = torch.arange(128, dtype=torch.int32, device="cuda")
    tile_source = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    output = torch.empty_like(source)
    tile_output = torch.empty_like(tile_source)
    kernel[(1, )](source, tile_source, output, tile_output, WAIT_TDM=wait_tdm, num_warps=4)
    if wait_tdm:
        torch.testing.assert_close(output, source)
        torch.testing.assert_close(tile_output, tile_source)


def async_copy_mbarrier_completion():
    mode = sys.argv[2]
    num_warps = int(sys.argv[3])

    @gluon.jit
    def kernel(source, output, tile_source, MODE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        block: ttgl.constexpr = 32 * num_warps
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])
        offsets = ttgl.arange(0, block, layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [2, block], ttgl.SwizzledSharedLayout(1, 1, 1, [0]))
        bars = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        if MODE != "uninitialized":
            ttgl.amd.cdna5.mbarrier.init(bars.index(0), count=2 * block if MODE == "two_arrivals" else block)
        ttgl.amd.cdna5.mbarrier.init(bars.index(1), count=block)
        if MODE == "empty":
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))
            ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
        elif MODE == "store_not_covered":
            smem.index(0).store(ttgl.load(source + offsets))
            ttgl.amd.cdna5.async_copy.shared_to_global(output + offsets, smem.index(0))
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))
            ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
            smem.index(0).store(ttgl.full([block], 0, ttgl.int32, layout))
        elif MODE == "tdm_not_covered":
            tile = tdm_load_tile(tile_source)
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))
            ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
            tile_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])
            tile.store(ttgl.full([32, 32], 0, ttgl.float16, tile_layout))
        else:
            if MODE == "masked":
                ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), source + offsets, mask=offsets < block // 2)
            else:
                ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), source + offsets)
            if MODE == "committed":
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))
            if MODE == "two_arrivals":
                ttgl.amd.cdna5.mbarrier.arrive(bars.index(0), count=1)
            elif MODE == "wrong_barrier":
                ttgl.amd.cdna5.mbarrier.arrive(bars.index(1), count=1)
            elif MODE == "after_arrive":
                ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(1), source + offsets)
            elif MODE == "reused_buffer":
                ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
                ttgl.amd.cdna5.async_copy.global_to_shared(smem.index(0), source + offsets)
            if MODE != "missing_wait":
                ttgl.amd.cdna5.mbarrier.wait(bars.index(1 if MODE == "wrong_barrier" else 0), phase=0)
            ttgl.store(output + offsets, smem.index(1 if MODE == "after_arrive" else 0).load(layout))

    block = 32 * num_warps
    source = torch.arange(block, dtype=torch.int32, device="cuda")
    output = torch.empty_like(source)
    tile_source = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    kernel[(1, )](source, output, tile_source, MODE=mode, num_warps=num_warps)
    if mode in ("wait", "committed", "two_arrivals"):
        torch.testing.assert_close(output, source)
    elif mode == "masked":
        torch.testing.assert_close(output[:block // 2], source[:block // 2])


def async_copy_mbarrier_warp_specialized():
    mode = sys.argv[2]
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def producer(source, output, smem, bars, layout: ttgl.constexpr, MODE: ttgl.constexpr):
        offsets = ttgl.arange(0, XBLOCK_C, layout)
        ttgl.amd.cdna5.async_copy.global_to_shared(smem, source + offsets)
        ttgl.amd.cdna5.async_copy.commit_group()
        if MODE == "group_before":
            ttgl.amd.cdna5.async_copy.wait_group(0)
        ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))
        ttgl.amd.cdna5.mbarrier.wait(bars.index(1), phase=0)
        if MODE == "group_after":
            for _ in range(130):
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.wait_group(0)
        ttgl.store(output + XBLOCK_C + offsets, smem.load(layout))

    @gluon.jit
    def consumer(output, smem, bars, layout: ttgl.constexpr):
        offsets = ttgl.arange(0, XBLOCK_C, layout)
        ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=0)
        ttgl.store(output + offsets, smem.load(layout))
        ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(1))

    @gluon.jit
    def handoff_producer(source, smem, bars):
        layout: ttgl.constexpr = ttgl.BlockedLayout([4], [32], [1], [0])
        offsets = ttgl.arange(0, XBLOCK_C, layout)
        for i in ttgl.static_range(2):
            if i > 0:
                ttgl.amd.cdna5.mbarrier.wait(bars.index(1), phase=i - 1)
            ttgl.amd.cdna5.async_copy.global_to_shared(smem, source + i * XBLOCK_C + offsets)
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(0))

    @gluon.jit
    def handoff_consumer(output, smem, bars, layout: ttgl.constexpr):
        offsets = ttgl.arange(0, XBLOCK_C, layout)
        for i in ttgl.static_range(2):
            ttgl.amd.cdna5.mbarrier.wait(bars.index(0), phase=i)
            ttgl.store(output + i * XBLOCK_C + offsets, smem.load(layout))
            ttgl.amd.cdna5.mbarrier.arrive(bars.index(1), count=1)

    @gluon.jit
    def kernel(source, output, MODE: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK_C], ttgl.SwizzledSharedLayout(1, 1, 1, [0]))
        bars = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        if MODE == "handoff":
            ttgl.amd.cdna5.mbarrier.init(bars.index(0), count=32)
            ttgl.amd.cdna5.mbarrier.init(bars.index(1), count=XBLOCK_C)
            ttgl.warp_specialize([
                (handoff_consumer, (output, smem, bars, layout)),
                (handoff_producer, (source, smem, bars)),
            ], [1])
        else:
            for i in ttgl.static_range(2):
                ttgl.amd.cdna5.mbarrier.init(bars.index(i), count=XBLOCK_C)
            ttgl.warp_specialize([
                (producer, (source, output, smem, bars, layout, MODE)),
                (consumer, (output, smem, bars, layout)),
            ], [4])

    source = torch.arange(2 * XBLOCK, dtype=torch.int32, device="cuda")
    output = torch.empty((2, XBLOCK), dtype=torch.int32, device="cuda")
    kernel[(1, )](source, output, MODE=mode, num_warps=4)
    if mode == "handoff":
        torch.testing.assert_close(output, source.view(2, XBLOCK))
    elif mode != "non_acquiring":
        torch.testing.assert_close(output, source[:XBLOCK].expand(2, XBLOCK))


def async_copy_mbarrier_storage():
    mode = sys.argv[2]
    num_warps = int(sys.argv[3])

    @gluon.jit
    def arriver(source, smem, bar, sync, layout: ttgl.constexpr, block: ttgl.constexpr):
        ttgl.amd.cdna5.async_copy.global_to_shared(smem, source + ttgl.arange(0, block, layout))
        ttgl.amd.cdna5.async_copy.mbarrier_arrive(bar)
        ttgl.amd.cdna5.mbarrier.arrive(sync, count=1)

    @gluon.jit
    def reinitializer(bar, sync, block: ttgl.constexpr):
        ttgl.amd.cdna5.mbarrier.wait(sync, phase=0)
        ttgl.amd.cdna5.async_copy.wait_group(-1)
        ttgl.amd.cdna5.mbarrier.init(bar, count=block)

    @gluon.jit
    def kernel(source, output, tile_source, MODE: ttgl.constexpr):
        num_warps: ttgl.constexpr = ttgl.num_warps()
        block: ttgl.constexpr = 32 * num_warps
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])
        offsets = ttgl.arange(0, block, layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [block], ttgl.SwizzledSharedLayout(1, 1, 1, [0]))
        bars = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        bar = bars.index(0)
        ttgl.amd.cdna5.mbarrier.init(bar, count=block)
        if MODE == "other_partition":
            ttgl.amd.cdna5.mbarrier.init(bars.index(1), count=block)
            ttgl.warp_specialize([
                (arriver, (source, smem, bar, bars.index(1), layout, block)),
                (reinitializer, (bar, bars.index(1), block)),
            ], [num_warps])
        else:
            if MODE == "tdm_after_wait_group_all":
                ttgl.amd.cdna5.mbarrier.init(bars.index(1), count=num_warps)
                tdm_load_tile(tile_source, mbarrier=bars.index(1))
            ttgl.amd.cdna5.async_copy.global_to_shared(smem, source + offsets)
            if MODE == "after_wait_group_zero_committed":
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bar)
            if MODE == "after_barrier_wait":
                ttgl.amd.cdna5.mbarrier.wait(bar, phase=0)
            elif MODE == "after_wait_group_zero_committed" or MODE == "after_wait_group_zero_uncommitted":
                ttgl.amd.cdna5.async_copy.wait_group(0)
            elif MODE != "pending":
                ttgl.amd.cdna5.async_copy.wait_group(-1)
            if MODE == "rearrive_after_wait_group_all":
                ttgl.amd.cdna5.async_copy.mbarrier_arrive(bar)
            ttgl.amd.cdna5.mbarrier.init(bars.index(1 if MODE == "tdm_after_wait_group_all" else 0), count=block)
            ttgl.store(output + offsets, smem.load(layout))

    block = 32 * num_warps
    source = torch.arange(block, dtype=torch.int32, device="cuda")
    output = torch.empty_like(source)
    tile_source = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    kernel[(1, )](source, output, tile_source, MODE=mode, num_warps=num_warps)
    if mode in ("after_barrier_wait", "after_wait_group_all", "after_wait_group_zero_committed"):
        torch.testing.assert_close(output, source)


def async_copy_mbarrier_pipeline_drain():
    mode = sys.argv[2]

    @gluon.jit
    def kernel(source, output, BLOCK: ttgl.constexpr, NUM_TILES: ttgl.constexpr, MODE: ttgl.constexpr):
        NUM_STAGES: ttgl.constexpr = 2
        COMMIT: ttgl.constexpr = MODE == "drain_zero"
        layout: ttgl.constexpr = ttgl.BlockedLayout([4], [32], [4], [0])
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [0])
        offsets = ttgl.arange(0, BLOCK, layout)
        ring = ttgl.allocate_shared_memory(ttgl.int8, [NUM_STAGES, BLOCK], shared_layout)
        bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_STAGES, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for k in ttgl.static_range(NUM_STAGES):
            ttgl.amd.cdna5.mbarrier.init(bars.index(k), count=128)
        for k in ttgl.static_range(NUM_STAGES):
            ttgl.amd.cdna5.async_copy.global_to_shared(ring.index(k), source + k * BLOCK + offsets)
            if COMMIT:
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(k))
        acc = ttgl.zeros([BLOCK], ttgl.int32, layout)
        for i in range(NUM_TILES):
            s = i % NUM_STAGES
            ttgl.amd.cdna5.mbarrier.wait(bars.index(s), (i // NUM_STAGES) & 1)
            acc += ring.index(s).load(layout).to(ttgl.int32)
            nxt = i + NUM_STAGES
            ttgl.amd.cdna5.async_copy.global_to_shared(ring.index(s), source + nxt * BLOCK + offsets, mask=offsets
                                                       < (NUM_TILES - nxt) * BLOCK)
            if COMMIT:
                ttgl.amd.cdna5.async_copy.commit_group()
            ttgl.amd.cdna5.async_copy.mbarrier_arrive(bars.index(s))
        if MODE == "drain_all":
            ttgl.amd.cdna5.async_copy.wait_group(-1)
        elif MODE == "drain_zero":
            ttgl.amd.cdna5.async_copy.wait_group(0)
        # The ring buffer and its barriers are dead here, so the epilogue tile can be allocated over them.
        # So the epilogue might reuse the barrier storage. ConSan must account for wait_group completing the async-copy barrier arrivals to avoid false reports.
        epilogue = ttgl.allocate_shared_memory(ttgl.int32, [BLOCK], shared_layout)
        epilogue.store(acc)
        ttgl.store(output + offsets, epilogue.load(layout))

    block, num_tiles = 512, 6
    source = (torch.arange(num_tiles * block, dtype=torch.int32) % 7).to(torch.int8).cuda()
    output = torch.empty(block, dtype=torch.int32, device="cuda")
    kernel[(1, )](source, output, BLOCK=block, NUM_TILES=num_tiles, MODE=mode, num_warps=4)
    if mode != "no_drain":
        torch.testing.assert_close(output, source.view(num_tiles, block).sum(0, dtype=torch.int32))


def tdm_store_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(output_ptr, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C, XBLOCK_C], smem_layout)
        val = ttgl.full([XBLOCK_C, XBLOCK_C], 42, ttgl.float16, blocked_layout)

        out_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)

        ttgl.amd.cdna5.tdm.async_store(out_desc, [0, 0], smem.index(0))
        ttgl.amd.cdna5.tdm.async_store(out_desc, [0, 0], smem.index(1))
        ttgl.amd.cdna5.tdm.async_wait(1)
        smem.index(0).store(val)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)
        smem.index(1).store(val)

    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


def tdm_load_no_barrier_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, out, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                         strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                         layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)

        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, val)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, FAILURE=FAILURE, num_warps=4)


def tdm_load_store_combined_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(input_ptr, output_ptr, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        in_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                            strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                            layout=smem_layout)
        out_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=in_desc.block_shape, layout=in_desc.layout)

        ttgl.amd.cdna5.tdm.async_load(in_desc, [0, 0], smem)
        ttgl.amd.cdna5.tdm.async_store(out_desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        ptr = output_ptr + out_m * XBLOCK_C + out_n
        ttgl.store(ptr, val)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, FAILURE=FAILURE, num_warps=4)


def tdm_two_bufs_one_wait_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def kernel(a_ptr, b_ptr, out, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[1, 0])

        a_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=a_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        b_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=b_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        a_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=a_desc.block_shape, layout=a_desc.layout)
        b_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=b_desc.block_shape, layout=b_desc.layout)

        ttgl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem)
        ttgl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(1)
        val = a_smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, val)

    a = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    b = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](a, b, output, FAILURE=FAILURE, num_warps=4)


def ws_store_wait_load_failure():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_consumer(smem, ready_bar, done_bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(ready_bar, phase=0)
        val = smem.index(0).load(layout)
        smem.index(1).store(val)
        ttgl.amd.cdna5.mbarrier.arrive(done_bar, count=1)

    @gluon.jit
    def ws_producer(smem, ready_bar, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        ttgl.amd.cdna5.mbarrier.arrive(ready_bar, count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for i in range(2):
            ttgl.amd.cdna5.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ready_bar = bar.index(0)
        done_bar = bar.index(1)
        ttgl.warp_specialize([
            (ws_consumer, (smem, ready_bar, done_bar, FAILURE, blocked_layout)),
            (ws_producer, (smem, ready_bar, XBLOCK_C, blocked_layout)),
        ], [4])
        ttgl.amd.cdna5.mbarrier.wait(done_bar, phase=0)
        val = smem.index(1).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK_C, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), dtype=torch.float16).cuda()
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


def ws_load_wait_store_failure():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        if not FAILURE:
            ttgl.amd.cdna5.mbarrier.wait(bar.index(0), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.cdna5.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        for i in range(2):
            ttgl.amd.cdna5.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4])
        ttgl.amd.cdna5.mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK_C, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), dtype=torch.float16).cuda()
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


def tdm_cross_partition_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)
        ttgl.amd.cdna5.mbarrier.arrive(sync_bar, count=1)

    @gluon.jit
    def ws_1(desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.cdna5.mbarrier.wait(sync_bar, phase=0)
        ttgl.amd.cdna5.tdm.async_load(desc, [0, 0], smem)
        ttgl.amd.cdna5.tdm.async_wait(0)

    @gluon.jit
    def kernel(input_ptr, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

        desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                         strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                         layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)
        sync_bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(sync_bar.index(0), count=NUM_WARPS * WARP_SIZE)

        ttgl.warp_specialize([
            (ws_default, (desc, smem, sync_bar.index(0), FAILURE)),
            (ws_1, (desc, smem, sync_bar.index(0), FAILURE)),
        ], [NUM_WARPS])

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, FAILURE=FAILURE, num_warps=4)


def tdm_cross_partition_load_store_kernel():
    FAILURE = sys.argv[2] == "True"
    XBLOCK_C: ttgl.constexpr = ttgl.constexpr(XBLOCK)

    @gluon.jit
    def ws_default(in_desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.cdna5.tdm.async_load(in_desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.cdna5.tdm.async_wait(0)
        ttgl.amd.cdna5.mbarrier.arrive(sync_bar, count=1)

    @gluon.jit
    def ws_1(out_desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.cdna5.mbarrier.wait(sync_bar, phase=0)
        ttgl.amd.cdna5.tdm.async_store(out_desc, [0, 0], smem)
        ttgl.amd.cdna5.tdm.async_wait(0)

    @gluon.jit
    def kernel(input_ptr, output_ptr, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

        in_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                            strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                            layout=smem_layout)
        out_desc = ttgl.amd.cdna5.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=in_desc.block_shape, layout=in_desc.layout)
        sync_bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.cdna5.mbarrier.MBarrierLayout())
        ttgl.amd.cdna5.mbarrier.init(sync_bar.index(0), count=NUM_WARPS * WARP_SIZE)

        ttgl.warp_specialize([
            (ws_default, (in_desc, smem, sync_bar.index(0), FAILURE)),
            (ws_1, (out_desc, smem, sync_bar.index(0), FAILURE)),
        ], [NUM_WARPS])

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    output = torch.empty((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, output, FAILURE=FAILURE, num_warps=4)


if __name__ == "__main__":
    tests = {
        "ws_store_wait_load_failure": ws_store_wait_load_failure, "ws_load_wait_store_failure":
        ws_load_wait_store_failure, "deadlock_two_partitions": deadlock_two_partitions, "deadlock_overarrival":
        deadlock_overarrival, "deadlock_underarrival": deadlock_underarrival, "deadlock_different_phases":
        deadlock_different_phases, "barrier_underflow": barrier_underflow, "aliasing_shared_visibility":
        aliasing_shared_visibility, "ws_two_loads_two_bars": ws_two_loads_two_bars, "ws_two_loads_one_bar":
        ws_two_loads_one_bar, "ws_two_loads_two_bars_loop": ws_two_loads_two_bars_loop, "ws_load_ordering":
        ws_load_ordering, "ws_different_warp_sizes": ws_different_warp_sizes, "async_tdm_kernel": async_tdm_kernel,
        "async_tdm_kernel_2bufs_1bar": async_tdm_kernel_2bufs_1bar, "tdm_interleave_kernel": tdm_interleave_kernel,
        "async_copy_kernel": async_copy_kernel, "tdm_store_kernel": tdm_store_kernel, "tdm_load_no_barrier_kernel":
        tdm_load_no_barrier_kernel, "tdm_load_store_combined_kernel": tdm_load_store_combined_kernel,
        "tdm_two_bufs_one_wait_kernel": tdm_two_bufs_one_wait_kernel, "tdm_cross_partition_kernel":
        tdm_cross_partition_kernel, "tdm_cross_partition_load_store_kernel": tdm_cross_partition_load_store_kernel
    }
    tests["barrier_storage_lifetime"] = barrier_storage_lifetime
    tests["barrier_storage_pending_completion"] = barrier_storage_pending_completion
    tests["async_wait_count_bounds"] = async_wait_count_bounds
    tests["async_wait_uncommitted"] = async_wait_uncommitted
    tests["async_copy_load_store"] = async_copy_load_store
    tests["async_copy_partition_isolation"] = async_copy_partition_isolation
    tests["async_copy_wait_pending_tdm"] = async_copy_wait_pending_tdm
    tests["async_copy_mbarrier_completion"] = async_copy_mbarrier_completion
    tests["async_copy_mbarrier_warp_specialized"] = async_copy_mbarrier_warp_specialized
    tests["async_copy_mbarrier_storage"] = async_copy_mbarrier_storage
    tests["async_copy_mbarrier_pipeline_drain"] = async_copy_mbarrier_pipeline_drain
    tests[sys.argv[1]]()
    torch.cuda.synchronize()
    print("ConSan helper completed", flush=True)
