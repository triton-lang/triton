# ruff: noqa: E402
import hip

hip.hip.hipInit(0)

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


def deadlock_two_partitions():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def deadlock_overarrival():

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

    kernel[(1, )](num_warps=4)


def deadlock_underarrival():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=8 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=8 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def deadlock_different_phases():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=1)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4])

    kernel[(1, )](num_warps=4)


def barrier_underflow():

    @gluon.jit
    def ws_default(bar):
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=3)
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        WARP_SIZE: ttgl.constexpr = 32
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
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
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def reader(alias1: ttgl.constexpr, dummy: ttgl.constexpr, bar: ttgl.constexpr, MISSING_BAR: ttgl.constexpr,
               blocked_layout: ttgl.constexpr):
        if not MISSING_BAR:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
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
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
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
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        if MISSING_BAR != "1":
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(3):
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4])
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(2), phase=0)
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
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        if not FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=2 * 4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, blocked_layout)),
            (ws_1, (smem, bar, blocked_layout)),
            (ws_2, (smem, bar, FAILURE, blocked_layout)),
        ], [4, 4])
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)
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
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
            acc = acc + val
        smem.index(1).store(acc)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK_C], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(3), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)
            acc = acc + val
        smem.index(2).store(acc)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "0":
                ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=phase)
            if MISSING_BAR != "1":
                ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(3), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [4, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(4):
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(3), count=1)
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
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
            smem.index(1).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK_C], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=phase)
            val = smem.index(1 if FAILURE else 0).load(layout)
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)
            acc = acc + val
        smem.index(2).store(acc)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(3):
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)
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
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        if MISSING_BAR != "1":
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(2), count=1)

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
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=4 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=2 * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(2), count=8 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, layout_4)),
            (ws_1, (smem, bar, MISSING_BAR, layout_2)),
            (ws_2, (smem, bar, MISSING_BAR, layout_8)),
        ], [2, 8])
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(2), phase=0)
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

        desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=NUM_WARPS)

        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem, mbarrier=bar.index(0))
        if not FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

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

        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        a_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=a_desc.block_shape, layout=a_desc.layout)
        b_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=b_desc.block_shape, layout=b_desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        # 2 TDM loads x NUM_WARPS per-warp arrivals = 2*NUM_WARPS total
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=2 * NUM_WARPS)

        ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_smem, mbarrier=bar.index(0))
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b_smem, mbarrier=bar.index(0))
        if not FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        val = a_smem.load(blocked_layout)
        val = val + b_smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)

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

        desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2] + list(desc.block_shape), desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(bar.index(0), count=NUM_WARPS)
        ttgl.amd.gfx1250.mbarrier.init(bar.index(1), count=NUM_WARPS)

        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem.index(0), mbarrier=bar.index(0))
        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem.index(1), mbarrier=bar.index(1))

        ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        if not FAILURE:
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)

        out_m = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK_C, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK_C + out_n
        ttgl.store(out_ptr, smem.index(0).load(blocked_layout))
        ttgl.store(out_ptr, smem.index(1).load(blocked_layout))

        out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                               strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                               layout=smem_layout)
        ttgl.amd.gfx1250.tdm.async_store(out_desc, [0, 0], smem.index(0))
        ttgl.amd.gfx1250.tdm.async_wait(0)

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
        ttgl.amd.gfx1250.async_copy.global_to_shared(smem.index(0), input_ptr + offs)
        ttgl.amd.gfx1250.async_copy.commit_group()

        ttgl.amd.gfx1250.async_copy.global_to_shared(smem.index(1), input_ptr + offs)
        ttgl.amd.gfx1250.async_copy.commit_group()
        ttgl.amd.gfx1250.async_copy.wait_group(2 if FAILURE else 1)

        ttgl.amd.gfx1250.async_copy.global_to_shared(smem.index(0), input_ptr + offs)
        ttgl.amd.gfx1250.async_copy.commit_group()
        ttgl.amd.gfx1250.async_copy.wait_group(0)

    input_t = torch.randn((XBLOCK, XBLOCK), dtype=torch.float16).cuda()
    kernel[(1, )](input_t, FAILURE=FAILURE, num_warps=4)


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

        out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                               strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                               layout=smem_layout)

        ttgl.amd.gfx1250.tdm.async_store(out_desc, [0, 0], smem.index(0))
        ttgl.amd.gfx1250.tdm.async_store(out_desc, [0, 0], smem.index(1))
        ttgl.amd.gfx1250.tdm.async_wait(1)
        smem.index(0).store(val)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)
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

        desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)

        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)

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

        in_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                              strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                              layout=smem_layout)
        out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                               strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                               layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=in_desc.block_shape, layout=in_desc.layout)

        ttgl.amd.gfx1250.tdm.async_load(in_desc, [0, 0], smem)
        ttgl.amd.gfx1250.tdm.async_store(out_desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)
        val = smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)

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

        a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                             strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                             layout=smem_layout)
        a_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=a_desc.block_shape, layout=a_desc.layout)
        b_smem = ttgl.allocate_shared_memory(ttgl.float16, shape=b_desc.block_shape, layout=b_desc.layout)

        ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_smem)
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b_smem)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(1)
        val = a_smem.load(blocked_layout)
        if FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)

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
            ttgl.amd.gfx1250.mbarrier.wait(ready_bar, phase=0)
        val = smem.index(0).load(layout)
        smem.index(1).store(val)
        ttgl.amd.gfx1250.mbarrier.arrive(done_bar, count=1)

    @gluon.jit
    def ws_producer(smem, ready_bar, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(ready_bar, count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(2):
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ready_bar = bar.index(0)
        done_bar = bar.index(1)
        ttgl.warp_specialize([
            (ws_consumer, (smem, ready_bar, done_bar, FAILURE, blocked_layout)),
            (ws_producer, (smem, ready_bar, XBLOCK_C, blocked_layout)),
        ], [4])
        ttgl.amd.gfx1250.mbarrier.wait(done_bar, phase=0)
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
            ttgl.amd.gfx1250.mbarrier.wait(bar.index(0), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK_C, layout).to(ttgl.float16))
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        ttgl.amd.gfx1250.mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK_C], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        for i in range(2):
            ttgl.amd.gfx1250.mbarrier.init(bar.index(i), count=4 * WARP_SIZE)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4])
        ttgl.amd.gfx1250.mbarrier.wait(bar.index(1), phase=0)
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
        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)
        ttgl.amd.gfx1250.mbarrier.arrive(sync_bar, count=1)

    @gluon.jit
    def ws_1(desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.gfx1250.mbarrier.wait(sync_bar, phase=0)
        ttgl.amd.gfx1250.tdm.async_load(desc, [0, 0], smem)
        ttgl.amd.gfx1250.tdm.async_wait(0)

    @gluon.jit
    def kernel(input_ptr, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

        desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                           strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                           layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=desc.block_shape, layout=desc.layout)
        sync_bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(sync_bar.index(0), count=NUM_WARPS * WARP_SIZE)

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
        ttgl.amd.gfx1250.tdm.async_load(in_desc, [0, 0], smem)
        if not FAILURE:
            ttgl.amd.gfx1250.tdm.async_wait(0)
        ttgl.amd.gfx1250.mbarrier.arrive(sync_bar, count=1)

    @gluon.jit
    def ws_1(out_desc, smem, sync_bar, FAILURE: ttgl.constexpr):
        ttgl.amd.gfx1250.mbarrier.wait(sync_bar, phase=0)
        ttgl.amd.gfx1250.tdm.async_store(out_desc, [0, 0], smem)
        ttgl.amd.gfx1250.tdm.async_wait(0)

    @gluon.jit
    def kernel(input_ptr, output_ptr, FAILURE: ttgl.constexpr):
        WARP_SIZE: ttgl.constexpr = 32
        NUM_WARPS: ttgl.constexpr = 4
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

        in_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=input_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                              strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                              layout=smem_layout)
        out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=output_ptr, shape=(XBLOCK_C, XBLOCK_C),
                                                               strides=(XBLOCK_C, 1), block_shape=(XBLOCK_C, XBLOCK_C),
                                                               layout=smem_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, shape=in_desc.block_shape, layout=in_desc.layout)
        sync_bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], ttgl.amd.gfx1250.mbarrier.MBarrierLayout())
        ttgl.amd.gfx1250.mbarrier.init(sync_bar.index(0), count=NUM_WARPS * WARP_SIZE)

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
    tests[sys.argv[1]]()
