"""
Dense FP4 Matrix Multiplication with K96
=======================================

MXFP4 (block32) and NVFP4 (block16), using unsigned packed FP4 operands,
FP32 accumulation, and FP16 output. Requires sm103 and K divisible by 768.
There is no operand padding: three exact K256 TMA stages feed eight K96 MMAs.

Scale packing and the MXFP4 pipeline are shared with example 04. NVFP4 uses
six data slots and five independently released scale slots. Its K128 scale
copies are interleaved with K96 MMAs without adding copies or transferred bytes.
Each data slot is released after its last consuming instruction. The NVFP4
epilogue drains the accumulator early and reuses retired input storage for the
last static-persistent output tile. Grouped traversal preserves L2 reuse;
cluster launch control remains available as an explicit scheduler choice.

Run a same-input comparison against an archived native K96 binary::

    python 07-pure-k96-matmul.py --size 16384 --k 16128 --format nvfp4 \
        --modes native --frozen-native /path/to/frozen/native \
        --repeats 7 --rep-ms 500 --output /tmp/dense-k96

The benchmark checks FP32 references before timing and saves individual samples,
PTX, cubins, compiler metadata, and source hashes. See K96_EXPLORATION.md for
measured results and rejected experiments; 9 PFLOPS is not an achieved result.
"""
import importlib.util
import sys
from pathlib import Path
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, TensorMemoryScalesLayout,
                                                                 allocate_tensor_memory, tcgen05_copy, tcgen05_commit,
                                                                 tcgen05_mma_barrier_count, tcgen05_mma_scaled,
                                                                 mbarrier, tma)

_base_path = Path(__file__).with_name('04-2cta-block-scale-matmul.py')
_spec = importlib.util.spec_from_file_location('dense_k96_base', _base_path)
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
def _copy_scales(p, idx, sa, sb, OFFSET: gl.constexpr, VEC: gl.constexpr, SRC: gl.constexpr = 0,
                 WIDTH: gl.constexpr = 256):
    a = unswizzle_scales_shared_memory(p.a_scale_bufs.index(idx), 256, 256, VEC)
    b = unswizzle_scales_shared_memory(p.b_scale_bufs.index(idx), 256, 256, VEC)
    tcgen05_copy(a.slice(SRC // VEC, WIDTH // VEC, 1), sa.slice(OFFSET // VEC, WIDTH // VEC))
    tcgen05_copy(b.slice(SRC // VEC, WIDTH // VEC, 1), sb.slice(OFFSET // VEC, WIDTH // VEC))


@gluon.jit
def _mma_partition(p, scale_empty):
    SPLIT_SCALES: gl.constexpr = p.a_scale_bufs.shape[0] != p.a_bufs.shape[0]
    COPY_K: gl.constexpr = 128 if SPLIT_SCALES else 256
    scale_state = Counter.create(0, p.a_scale_bufs.shape[0])
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
            q0 = scale_state if SPLIT_SCALES else load_state
            q1 = q0.next()
            q2 = q1.next()
            s0 = load_state
            s1 = s0.next()
            s2 = s1.next()
            a0, b0 = p.a_bufs.index(s0.index), p.b_bufs.index(s0.index).permute((1, 0))
            a1, b1 = p.a_bufs.index(s1.index), p.b_bufs.index(s1.index).permute((1, 0))
            a2, b2 = p.a_bufs.index(s2.index), p.b_bufs.index(s2.index).permute((1, 0))
            mbarrier.wait(p.load_ready_bars.index(s0.index), s0.phase)
            _copy_scales(p, q0.index, sa, sb, 0, VEC, WIDTH=COPY_K)
            if SPLIT_SCALES:
                tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", use_acc=use_acc, k_range=(0, 96),
                                   instruction_k=96, scale_block_size=VEC, a_scale_offset=0, b_scale_offset=0,
                                   multicast=True, mbarriers=[], is_async=True)
                _copy_scales(p, q0.index, sa, sb, 128, VEC, SRC=128, WIDTH=128)
                tcgen05_commit(scale_empty.index(q0.index),
                               descs=[p.a_scale_bufs.index(q0.index),
                                      p.b_scale_bufs.index(q0.index)])
                tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", k_range=(96, 192), instruction_k=96,
                                   scale_block_size=VEC, a_scale_offset=96 // VEC, b_scale_offset=96 // VEC,
                                   multicast=True, mbarriers=[], is_async=True)
            else:
                tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", use_acc=use_acc, k_range=(0, 192),
                                   instruction_k=96, scale_block_size=VEC, a_scale_offset=0, b_scale_offset=0,
                                   multicast=True, mbarriers=[], is_async=True)
            mbarrier.wait(p.load_ready_bars.index(s1.index), s1.phase)
            _copy_scales(p, q1.index, sa, sb, 256, VEC, WIDTH=COPY_K)
            tcgen05_mma_scaled(a0, b0, acc, sa, sb, "e2m1", "e2m1", a_next=a1, b_next=b1, k_range=(192, 288),
                               instruction_k=96, scale_block_size=VEC, a_scale_offset=192 // VEC,
                               b_scale_offset=192 // VEC, multicast=True, mbarriers=[p.load_empty_bars.index(s0.index)])
            if SPLIT_SCALES:
                tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1", k_range=(32, 128), instruction_k=96,
                                   scale_block_size=VEC, a_scale_offset=288 // VEC, b_scale_offset=288 // VEC,
                                   multicast=True, mbarriers=[], is_async=True)
                _copy_scales(p, q1.index, sa, sb, 384, VEC, SRC=128, WIDTH=128)
                tcgen05_commit(scale_empty.index(q1.index),
                               descs=[p.a_scale_bufs.index(q1.index),
                                      p.b_scale_bufs.index(q1.index)])
                tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1", k_range=(128, 224), instruction_k=96,
                                   scale_block_size=VEC, a_scale_offset=384 // VEC, b_scale_offset=384 // VEC,
                                   multicast=True, mbarriers=[], is_async=True)
            else:
                tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1", k_range=(32, 224), instruction_k=96,
                                   scale_block_size=VEC, a_scale_offset=288 // VEC, b_scale_offset=288 // VEC,
                                   multicast=True, mbarriers=[], is_async=True)
            mbarrier.wait(p.load_ready_bars.index(s2.index), s2.phase)
            _copy_scales(p, q2.index, sa, sb, 512, VEC, WIDTH=COPY_K)
            tcgen05_mma_scaled(a1, b1, acc, sa, sb, "e2m1", "e2m1", a_next=a2, b_next=b2, k_range=(224, 320),
                               instruction_k=96, scale_block_size=VEC, a_scale_offset=480 // VEC,
                               b_scale_offset=480 // VEC, multicast=True, mbarriers=[p.load_empty_bars.index(s1.index)])
            if SPLIT_SCALES:
                _copy_scales(p, q2.index, sa, sb, 640, VEC, SRC=128, WIDTH=128)
                tcgen05_commit(scale_empty.index(q2.index),
                               descs=[p.a_scale_bufs.index(q2.index),
                                      p.b_scale_bufs.index(q2.index)])
            tcgen05_mma_scaled(a2, b2, acc, sa, sb, "e2m1", "e2m1", k_range=(64, 256), instruction_k=96,
                               scale_block_size=VEC, a_scale_offset=576 // VEC, b_scale_offset=576 // VEC,
                               multicast=True, mbarriers=[p.load_empty_bars.index(s2.index)])
            scale_state = q2.next()
            load_state = s2.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _load_partition(p, scale_empty):
    K = p.a_desc.shape[1] * 2
    data = Counter.create(1, p.load_empty_bars.shape[0])
    scales = Counter.create(1, p.a_scale_bufs.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        for k in range(0, K, 256):
            mbarrier.wait(p.load_empty_bars.index(data.index), data.phase)
            bar = p.load_ready_bars.index(data.index)
            mbarrier.expect(
                bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta + p.a_scale_desc.nbytes_per_cta +
                p.b_scale_desc.nbytes_per_cta)
            tma.async_load(p.a_desc, [off_m, k // 2], bar, p.a_bufs.index(data.index), multicast=True)
            tma.async_load(p.b_desc, [off_n, k // 2], bar, p.b_bufs.index(data.index), multicast=True)
            # Data has six slots. Only the independently staged scales wait
            # for their five-slot ring, after both data transfers are issued.
            mbarrier.wait(scale_empty.index(scales.index), scales.phase)
            sk = k // 256 * p.a_scale_desc.block_shape[2]
            tma.async_load(p.b_scale_desc, [0, scheduler.pid_n * 2, sk, 0, 0], bar, p.b_scale_bufs.index(scales.index),
                           multicast=True)
            tma.async_load(p.a_scale_desc, [0, scheduler.pid_m * 2, sk, 0, 0], bar, p.a_scale_bufs.index(scales.index),
                           multicast=True)
            data = data.next()
            scales = scales.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _epilogue_partition(p, c_final):
    EP: gl.constexpr = p.c_desc.block_shape[1]
    buf = gl.allocate_shared_memory(p.c_desc.dtype, [256, EP], p.c_desc.layout)
    state = Counter.create(0, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)
        acc = p.acc_bufs.index(state.index)
        # Convert each half before loading the next, keeping the complete
        # output in FP16 registers without a 256-register FP32 live range.
        parts = ()
        for j in gl.static_range(2):
            chunks = (acc.slice(j * 128, 128).load().to(p.c_desc.dtype), )
            for level in gl.static_range((128 // EP).bit_length() - 1):
                next_chunks = ()
                for n in gl.static_range(len(chunks)):
                    left, right = chunks[n].reshape((256, 2, chunks[n].shape[1] // 2)).permute((0, 2, 1)).split()
                    next_chunks += (left, right)
                chunks = next_chunks
            parts += chunks
        mbarrier.arrive(p.acc_empty_bars.index(state.index), count=1)
        if p.scheduler == SCHEDULER_SPS:
            last = scheduler.tile_id + gl.num_programs(0) >= p.NUM_PID_M * p.NUM_PID_N
        else:
            last = False
        if last:
            tiles = parts
            for level in gl.static_range((256 // EP).bit_length() - 1):
                joined = ()
                for j in gl.static_range(len(tiles) // 2):
                    first, second = tiles[2 * j], tiles[2 * j + 1]
                    joined += (gl.join(first, second).permute((0, 2, 1)).reshape((256, first.shape[1] * 2)), )
                tiles = joined
            # All input reads have completed, and this persistent CTA has no
            # future producer work. Its retired input storage holds one wide
            # final collective store, avoiding the narrow epilogue's tail.
            scratch = p.a_bufs.slice(0, 4, 0).reinterpret(p.c_desc.dtype, [256, 256], c_final.layout)
            scratch.store(tiles[0])
            tma.async_store(c_final, [off_m, off_n], scratch)
        else:
            for j in gl.static_range(256 // EP):
                tma.store_wait(0)
                buf.store(parts[j])
                tma.async_store(p.c_desc, [off_m, off_n + j * EP], buf)
        state = state.next()
        scheduler = scheduler.step(i)
        i += 1
    tma.store_wait(0)


@gluon.jit
def dense_k96_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                     num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                     EPILOGUE_BLOCK_N: gl.constexpr, num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr,
                     GRID_TILE_WIDTH: gl.constexpr, CGA_LAYOUT: gl.constexpr, scheduler: gl.constexpr,
                     SCALE_BUFFERS: gl.constexpr, c_final):
    SPLIT_SCALES: gl.constexpr = SCALE_BUFFERS != num_buffers
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [SCALE_BUFFERS] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [SCALE_BUFFERS] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count([
        a_bufs.index(0),
        b_bufs.index(0),
        a_bufs.index(1),
        b_bufs.index(1),
        a_scale_bufs.index(0),
        b_scale_bufs.index(0)
    ], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    if SPLIT_SCALES:
        scale_empty = mbarrier.allocate_mbarrier(batch=SCALE_BUFFERS)
        for slot in gl.static_range(SCALE_BUFFERS):
            mbarrier.init(scale_empty.index(slot), count=mma_barrier_count)
    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    if not SPLIT_SCALES:
        scale_empty = load_empty_bars
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

    if SPLIT_SCALES:
        if scheduler == SCHEDULER_CLC:
            gl.warp_specialize([
                (_epilogue_partition, (p, c_final)),
                (_mma_partition, (p, scale_empty)),
                (_load_partition, (p, scale_empty)),
                (mma_scaled_clc_partition, (p, )),
            ], [1, 1, 1], [48, 48, 24])
        else:
            gl.warp_specialize([
                (_epilogue_partition, (p, c_final)),
                (_mma_partition, (p, scale_empty)),
                (_load_partition, (p, scale_empty)),
            ], [1, 1], [48, 48])
    else:
        if scheduler == SCHEDULER_CLC:
            gl.warp_specialize([
                (mma_scaled_epilogue_partition, (p, )),
                (_mma_partition, (p, scale_empty)),
                (mma_scaled_load_partition, (p, )),
                (mma_scaled_clc_partition, (p, )),
            ], [1, 1, 1], [24, 24, 24])
        else:
            gl.warp_specialize([
                (mma_scaled_epilogue_partition, (p, )),
                (_mma_partition, (p, scale_empty)),
                (mma_scaled_load_partition, (p, )),
            ], [1, 1], [24, 24])


def matmul(A, B, A_scale, B_scale, vec, *, buffers=None, epilogue=None, scheduler=None, out_dtype=torch.float16,
           tile_width=None):
    """Return (output, selected binary) for packed FP4 on sm103.

    Scales use example 04's packed block layout. Each K768 macrotile
    issues eight K96 instructions; NVFP4 interleaves K128 scale copies.
    Producer transfers remain exactly K256 per slot, with no operand padding
    or extra copies. The six-slot NVFP4 path uses 16-bit output; wider
    NVFP4 output retains the coupled five-slot pipeline and the same output
    staging byte budget.
    """
    assert torch.cuda.get_device_capability(A.device) == (10, 3)
    assert A.dtype == B.dtype == torch.uint8
    assert vec in (16, 32)
    if buffers is None:
        buffers = 6 if vec == 32 or out_dtype.itemsize == 2 else 5
    scale_buffers = min(buffers, 5) if vec == 16 else buffers
    if epilogue is None:
        epilogue = 16 if scale_buffers != buffers else (64 if vec == 32 else 128) // out_dtype.itemsize
    M, N, K = A.shape[0], B.shape[0], A.shape[1] * 2
    assert B.shape[1] * 2 == K and K % 768 == 0
    if scheduler is None:
        scheduler = SCHEDULER_SPS
    if tile_width is None:
        tile_width = 16 if K <= 16384 else 8
    ad, bd, cd, asd, bsd = base.make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N)
    base.mma_scaled_tma_set_block_size_hook(
        dict(a_desc=ad, b_desc=bd, c_desc=cd, a_scale_desc=asd, b_scale_desc=bsd, BLOCK_M=256, BLOCK_N=256, BLOCK_K=256,
             EPILOGUE_BLOCK_N=epilogue, CGA_LAYOUT=((1, 0), )))
    c_dtype = getattr(gl, str(out_dtype).split('.')[1])
    final_layout = gl.NVMMASharedLayout.get_default_for([256, 256], c_dtype, cga_layout=((1, 0), ))
    c_final = base.TensorDescriptor.from_tensor(cd.base, [256, 256], final_layout)
    grid = base.mma_scaled_warp_specialized_grid(M, N, 256, 256, 2, scheduler, A.device)
    compiled = dense_k96_kernel[grid](ad, bd, cd, asd, bsd, M, N, K, 2, buffers, 256, 256, 256, epilogue, 1, 0,
                                      tile_width, ((1, 0), ), scheduler, scale_buffers, c_final, num_ctas=2)
    return cd.base, compiled


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("dense_k96_benchmark",
                                                  Path(__file__).with_name("bench-tcgen05-pure-k96.py"))
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.main(default_example=Path(__file__).name)
