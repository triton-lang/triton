"""Native all-K96 Gluon matmul: eight K96 instructions per three packed K256 TMA slots."""
import importlib.util
import sys
from pathlib import Path
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, TensorMemoryScalesLayout,
                                                                 allocate_tensor_memory, tcgen05_copy, tcgen05_commit,
                                                                 tcgen05_mma_barrier_count, tcgen05_mma_scaled, mbarrier, tma)

_base_path = Path(__file__).with_name('04-2cta-block-scale-matmul.py')
_spec = importlib.util.spec_from_file_location('k96_base', _base_path)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
Counter = base.Counter
PartitionArgs = base.PartitionArgs
SCHEDULER_CLC = base.SCHEDULER_CLC
SCHEDULER_SPS = base.SCHEDULER_SPS
mma_scaled_load_partition = base.mma_scaled_load_partition
mma_scaled_epilogue_partition = base.mma_scaled_epilogue_partition
mma_scaled_clc_partition = base.mma_scaled_clc_partition
unswizzle_scales_shared_memory = base.unswizzle_scales_shared_memory


@gluon.jit
def copy_scales(p, idx, sa, sb, OFFSET: gl.constexpr, VEC: gl.constexpr):
    a = unswizzle_scales_shared_memory(p.a_scale_bufs.index(idx), 256, 256, VEC)
    b = unswizzle_scales_shared_memory(p.b_scale_bufs.index(idx), 256, 256, VEC)
    tcgen05_copy(a, sa.slice(OFFSET, 256 // VEC))
    tcgen05_copy(b, sb.slice(OFFSET, 256 // VEC))


@gluon.jit
def native_mma_partition(p):
    K = p.a_desc.shape[1] * 2
    VEC: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    sa = allocate_tensor_memory(p.a_scale_desc.dtype, [256, 1024 // VEC], TensorMemoryScalesLayout([[1, 0]]))
    sb = allocate_tensor_memory(p.b_scale_desc.dtype, [256, 1024 // VEC], TensorMemoryScalesLayout([[0, 0]]))
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, 768):
            s0 = load_state
            s1 = s0.next()
            s2 = s1.next()
            a0, b0 = p.a_bufs.index(s0.index), p.b_bufs.index(s0.index).permute((1, 0))
            a1, b1 = p.a_bufs.index(s1.index), p.b_bufs.index(s1.index).permute((1, 0))
            a2, b2 = p.a_bufs.index(s2.index), p.b_bufs.index(s2.index).permute((1, 0))
            mbarrier.wait(p.load_ready_bars.index(s0.index), s0.phase)
            copy_scales(p, s0.index, sa, sb, 0, VEC)
            tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", use_acc=use_acc,
                               k_range=(0, 192), instruction_k=96, scale_block_size=VEC,
                               a_scale_offset=0, b_scale_offset=0, multicast=True, mbarriers=[])
            mbarrier.wait(p.load_ready_bars.index(s1.index), s1.phase)
            copy_scales(p, s1.index, sa, sb, 256 // VEC, VEC)
            tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", a_next=a1, b_next=b1,
                               k_range=(192, 288), instruction_k=96, scale_block_size=VEC,
                               a_scale_offset=192 // VEC, b_scale_offset=192 // VEC,
                               multicast=True, mbarriers=[p.load_empty_bars.index(s0.index)])
            tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1",
                               k_range=(32, 224), instruction_k=96, scale_block_size=VEC,
                               a_scale_offset=288 // VEC, b_scale_offset=288 // VEC, multicast=True, mbarriers=[])
            mbarrier.wait(p.load_ready_bars.index(s2.index), s2.phase)
            copy_scales(p, s2.index, sa, sb, 512 // VEC, VEC)
            tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1", a_next=a2, b_next=b2,
                               k_range=(224, 320), instruction_k=96, scale_block_size=VEC,
                               a_scale_offset=480 // VEC, b_scale_offset=480 // VEC,
                               multicast=True, mbarriers=[p.load_empty_bars.index(s1.index)])
            tcgen05_mma_scaled(a2, b2, acc, sa, sb, "e2m1", "e2m1",
                               k_range=(64, 256), instruction_k=96, scale_block_size=VEC,
                               a_scale_offset=576 // VEC, b_scale_offset=576 // VEC,
                               multicast=True, mbarriers=[p.load_empty_bars.index(s2.index)])
            load_state = s2.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def pure_k96_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                    num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                    EPILOGUE_BLOCK_N: gl.constexpr, num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr,
                    GRID_TILE_WIDTH: gl.constexpr, CGA_LAYOUT: gl.constexpr, scheduler: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0),
         a_scale_bufs.index(0), b_scale_bufs.index(0)], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    clc_barriers = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    cga_layout_clc: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout_clc)
    clc_consumed_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], clc_layout)
    if scheduler == SCHEDULER_CLC:
        for i in gl.static_range(num_acc_buffers):
            mbarrier.init(clc_barriers.index(i), count=1)
            mbarrier.init(clc_planar_ready_bars.index(i), count=1)
            mbarrier.init(clc_consumed_bars.index(i), count=N_PARTITIONS - 1)

    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, clc_result_buffers,
                      clc_barriers, clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars, GRID_MINOR_DIM,
                      GRID_TILE_WIDTH, gl.cdiv(M, BLOCK_M), gl.cdiv(N, BLOCK_N), scheduler)

    if scheduler == SCHEDULER_CLC:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (native_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
            (mma_scaled_clc_partition, (p, )),
        ], [1, 1, 1], [24, 24, 24])
    else:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (native_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
        ], [1, 1], [24, 24])


def matmul(A, B, A_scale, B_scale, vec, *, buffers=None, epilogue=None, scheduler=None, out_dtype=torch.float16):
    """Return (output, selected binary) for packed FP4 on sm103.

    Scales use example 04's packed block layout. Five typed MMA operations
    consume each K768 macrotile using only K96 instructions. Producer transfers
    remain exactly K256 per slot, with no operand padding or extra copies.
    """
    assert torch.cuda.get_device_capability(A.device) == (10, 3)
    assert A.dtype == B.dtype == torch.uint8
    assert vec in (16, 32)
    if buffers is None:
        buffers = 6 if vec == 32 else 5
    if epilogue is None:
        epilogue = 32 if vec == 32 else 64
    M, N, K = A.shape[0], B.shape[0], A.shape[1] * 2
    assert B.shape[1] * 2 == K and K % 768 == 0
    if scheduler is None:
        scheduler = SCHEDULER_SPS if K <= 8192 else SCHEDULER_CLC
    ad, bd, cd, asd, bsd = base.make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N)
    base.mma_scaled_tma_set_block_size_hook(
        dict(a_desc=ad, b_desc=bd, c_desc=cd, a_scale_desc=asd, b_scale_desc=bsd, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256,
             EPILOGUE_BLOCK_N=epilogue, CGA_LAYOUT=((1, 0), )))
    grid = base.mma_scaled_warp_specialized_grid(M, N, 256, 256, 2, scheduler, A.device)
    compiled = pure_k96_kernel[grid](ad, bd, cd, asd, bsd, M, N, K, 2, buffers, 256, 256, 256, epilogue, 1, 0, 8,
                                     ((1, 0), ), scheduler, num_ctas=2)
    return cd.base, compiled
