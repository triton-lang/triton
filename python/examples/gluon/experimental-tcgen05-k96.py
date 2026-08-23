"""Experimental all-K96 Gluon pipeline; explicit PTX descriptors, no operand padding."""
import importlib.util
import sys
from pathlib import Path
import torch
import triton.backends.nvidia.driver as _driver
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language._core import builtin
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


@builtin
def address(desc, _semantic=None):
    return gl.tensor(_semantic.builder.create_experimental_descriptor_address(desc.handle), gl.uint32)


@gluon.jit
def smem_descriptor(ptr, next_ptr, BYTE_OFFSET: gl.constexpr, WIDTH: gl.constexpr):
    bits: gl.constexpr = (1 << 62) | (1 << 46) | (64 << 32)
    desc = (((ptr >> 4) + BYTE_OFFSET // 16) & 0x3fff).to(gl.uint64) | bits
    if BYTE_OFFSET + WIDTH // 2 > 128:
        desc = desc | (1 << 52) | (((next_ptr.to(gl.uint64) >> 4) & 0x3fff) << 16)
    return desc


@gluon.jit
def copy_scales(p, idx, sa, sb, OFFSET: gl.constexpr, VEC: gl.constexpr):
    a = unswizzle_scales_shared_memory(p.a_scale_bufs.index(idx), 256, 256, VEC)
    b = unswizzle_scales_shared_memory(p.b_scale_bufs.index(idx), 256, 256, VEC)
    tcgen05_copy(a, sa.slice(OFFSET, 256 // VEC))
    tcgen05_copy(b, sb.slice(OFFSET, 256 // VEC))


@gluon.jit
def experimental_load_partition(p, MODE: gl.constexpr):
    if MODE != 'exact384' and MODE != 'exact192':
        mma_scaled_load_partition(p)
    else:
        state = Counter.create(1, p.load_empty_bars.shape[0])
        if p.scheduler == SCHEDULER_CLC:
            scheduler = p.get_clc_consumer()
        else:
            scheduler = p.get_sps_scheduler()
        VEC: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
        i = 0
        while scheduler.has_work:
            for k in range(0, p.a_desc.shape[1] * 2, 384):
                if MODE == 'exact192':
                    for j in gl.static_range(2):
                        idx = state.index
                        mbarrier.wait(p.load_empty_bars.index(idx), state.phase)
                        bar = p.load_ready_bars.index(idx)
                        mbarrier.expect(
                            bar, (p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta) * 3 // 4 +
                            ((p.a_scale_desc.nbytes_per_cta + p.b_scale_desc.nbytes_per_cta) * 3 // 4 if j == 0 else 0))
                        tma.async_load(p.a_desc, [scheduler.pid_m * 256, k // 2 + j * 96], bar, p.a_bufs.index(idx),
                                       multicast=True)
                        tma.async_load(p.b_desc, [scheduler.pid_n * 256, k // 2 + j * 96], bar, p.b_bufs.index(idx),
                                       multicast=True)
                        if j == 0:
                            tma.async_load(p.a_scale_desc, [0, scheduler.pid_m * 2, k // (VEC * 4), 0, 0], bar,
                                           p.a_scale_bufs.index(idx // 2), multicast=True)
                            tma.async_load(p.b_scale_desc, [0, scheduler.pid_n * 2, k // (VEC * 4), 0, 0], bar,
                                           p.b_scale_bufs.index(idx // 2), multicast=True)
                        state = state.next()
                else:
                    idx = state.index
                    mbarrier.wait(p.load_empty_bars.index(idx), state.phase)
                    bar = p.load_ready_bars.index(idx)
                    mbarrier.expect(bar, (p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta) * 3 // 2 +
                                    (p.a_scale_desc.nbytes_per_cta + p.b_scale_desc.nbytes_per_cta) * 3 // 4)
                    for j in gl.static_range(2):
                        tma.async_load(p.a_desc, [scheduler.pid_m * 256, k // 2 + j * 96], bar,
                                       p.a_bufs.index(2 * idx + j), multicast=True)
                        tma.async_load(p.b_desc, [scheduler.pid_n * 256, k // 2 + j * 96], bar,
                                       p.b_bufs.index(2 * idx + j), multicast=True)
                    tma.async_load(p.a_scale_desc, [0, scheduler.pid_m * 2, k // (VEC * 4), 0, 0], bar,
                                   p.a_scale_bufs.index(idx), multicast=True)
                    tma.async_load(p.b_scale_desc, [0, scheduler.pid_n * 2, k // (VEC * 4), 0, 0], bar,
                                   p.b_scale_bufs.index(idx), multicast=True)
                    state = state.next()
            scheduler = scheduler.step(i)
            i += 1


@gluon.constexpr_function
def group_asm(count, vec, commit_after):
    op = 'tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32' if vec == 32 else 'tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16'
    text = '{ bar.warp.sync 0xffffffff; .reg .pred e,p,u; .reg .b32 c; .reg .b16 mask; mov.b16 mask,3; elect.sync _|p,0xffffffff; setp.ne.u32 u,$2,0; '
    for i in range(count):
        k = 4 + 5 * i
        text += f'@p {op} [$1], ${k}, ${k+1}, ${k+2}, [${k+3}], [${k+4}], u; mov.pred u,1; '
        if i == commit_after:
            text += '@p tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$3],mask; '
    return text + 'mov.u32 $0,0; }'


@gluon.constexpr_function
def inst_bits(width, vec, offset):
    sid = offset % 4
    return 0x10400480 | (0x800000 if vec == 32 else 0) | ((1 << 31) if width == 96 else 0) | (sid << 29) | (sid << 4)


@gluon.jit
def raw_group(aa, bb, na, nb, acc, sa, sb, use_acc, commit, OFFSETS: gl.constexpr, SCALES: gl.constexpr,
              WIDTHS: gl.constexpr, VEC: gl.constexpr, COMMIT_AFTER: gl.constexpr = -1):
    lead = gl.inline_asm_elementwise('mov.u32 $0,%cluster_ctarank;', '=r', [], dtype=gl.int32, is_pure=True,
                                     pack=1) == 0
    if lead:
        args = (acc, gl.to_tensor(use_acc).to(gl.int32), gl.to_tensor(commit).to(gl.uint32))
        for i in gl.static_range(len(OFFSETS)):
            da = smem_descriptor(aa[i], na[i], OFFSETS[i], WIDTHS[i])
            db = smem_descriptor(bb[i], nb[i], OFFSETS[i], WIDTHS[i])
            args += (da, db, gl.to_tensor(inst_bits(WIDTHS[i], VEC, SCALES[i])).to(gl.uint32),
                     sa + 4 * (SCALES[i] // 4), sb + 8 * (SCALES[i] // 4))
        gl.inline_asm_elementwise(group_asm(len(OFFSETS), VEC, COMMIT_AFTER), '=r,r,r,r' + ',l,l,r,r,r' * len(OFFSETS),
                                  args, dtype=gl.int32, is_pure=False, pack=1)


@gluon.constexpr_function
def scale_group_asm(words):
    text = '{ .reg .pred e,p; .reg .b32 c; mov.u32 c,%cluster_ctarank; setp.eq.u32 p,c,0; elect.sync _|e,0xffffffff; and.pred p,p,e; '
    for i in range(words * 3):
        text += f'@p tcgen05.cp.cta_group::2.warpx4.32x128b [${1+2*i}],${2+2*i}; '
    return text + 'mov.u32 $0,0; }'


@gluon.jit
def grouped_exact_scale_copy(p, idx, sa, sb, VEC: gl.constexpr):
    a, b = address(p.a_scale_bufs.index(idx)), address(p.b_scale_bufs.index(idx))
    args = ()
    for word in gl.static_range(384 // (4 * VEC)):
        for mn in gl.static_range(3):
            src = a + word * 512 if mn == 0 else b + (word + (mn - 1) * (384 // (4 * VEC))) * 512
            dst = sa + word * 4 if mn == 0 else sb + word * 8 + (mn - 1) * 4
            desc = ((src >> 4) & 0x3fff).to(gl.uint64) | (1 << 46) | (8 << 32)
            args += (dst, desc)
    gl.inline_asm_elementwise(scale_group_asm(384 // (4 * VEC)), '=r' + ',r,l' * (3 * 384 // (4 * VEC)), args,
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def raw_mma_partition(p, MODE: gl.constexpr):
    K = p.a_desc.shape[1] * 2
    VEC: gl.constexpr = 32 if p.a_scale_desc.dtype == gl.uint8 else 16
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    SCALE_K: gl.constexpr = (1024 if MODE == 'pure' else
                             (512 if MODE == 'exact384' or MODE == 'exact192' else 256)) // VEC
    sa = allocate_tensor_memory(p.a_scale_desc.dtype, [256, SCALE_K], TensorMemoryScalesLayout([[1, 0]]))
    sb = allocate_tensor_memory(p.b_scale_desc.dtype, [256, SCALE_K], TensorMemoryScalesLayout([[0, 0]]))
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        acc = address(acc_buf)
        sfa, sfb = address(sa), address(sb)
        use_acc = False
        if MODE == 'pure':
            for k in range(0, K, 768):
                s0 = load_state
                s1 = s0.next()
                s2 = s1.next()
                a0, b0 = address(p.a_bufs.index(s0.index)), address(p.b_bufs.index(s0.index))
                a1, b1 = address(p.a_bufs.index(s1.index)), address(p.b_bufs.index(s1.index))
                a2, b2 = address(p.a_bufs.index(s2.index)), address(p.b_bufs.index(s2.index))
                mbarrier.wait(p.load_ready_bars.index(s0.index), s0.phase)
                copy_scales(p, s0.index, sa, sb, 0, VEC)
                raw_group((a0, a0), (b0, b0), (a0, a0), (b0, b0), acc, sfa, sfb, use_acc, 0, (0, 48), (0, 96 // VEC),
                          (96, 96), VEC)
                mbarrier.wait(p.load_ready_bars.index(s1.index), s1.phase)
                copy_scales(p, s1.index, sa, sb, 256 // VEC, VEC)
                raw_group((a0, a1, a1), (b0, b1, b1), (a1, a1, a1), (b1, b1, b1), acc, sfa, sfb, True,
                          address(p.load_empty_bars.index(s0.index)), (96, 16, 64),
                          (192 // VEC, 288 // VEC, 384 // VEC), (96, 96, 96), VEC, COMMIT_AFTER=0)
                mbarrier.wait(p.load_ready_bars.index(s2.index), s2.phase)
                copy_scales(p, s2.index, sa, sb, 512 // VEC, VEC)
                raw_group((a1, a2, a2), (b1, b2, b2), (a2, a2, a2), (b2, b2, b2), acc, sfa, sfb, True,
                          address(p.load_empty_bars.index(s1.index)), (112, 32, 80),
                          (480 // VEC, 576 // VEC, 672 // VEC), (96, 96, 96), VEC, COMMIT_AFTER=0)
                tcgen05_commit(
                    p.load_empty_bars.index(s2.index), descs=[
                        p.a_bufs.index(s2.index),
                        p.b_bufs.index(s2.index),
                        p.a_scale_bufs.index(s2.index),
                        p.b_scale_bufs.index(s2.index)
                    ])
                load_state = s2.next()
                use_acc = True
        elif MODE == 'exact192':
            for k in range(0, K, 384):
                for j in gl.static_range(2):
                    idx = load_state.index
                    mbarrier.wait(p.load_ready_bars.index(idx), load_state.phase)
                    if j == 0:
                        grouped_exact_scale_copy(p, idx // 2, sfa, sfb, VEC)
                    a, b = address(p.a_bufs.index(idx)), address(p.b_bufs.index(idx))
                    raw_group((a, a), (b, b), (a, a), (b, b), acc, sfa, sfb, use_acc,
                              address(p.load_empty_bars.index(idx)), (0, 48), (j * 192 // VEC, (j * 192 + 96) // VEC),
                              (96, 96), VEC, COMMIT_AFTER=1)
                    use_acc = True
                    load_state = load_state.next()
        elif MODE == 'exact384':
            for k in range(0, K, 384):
                idx = load_state.index
                mbarrier.wait(p.load_ready_bars.index(idx), load_state.phase)
                grouped_exact_scale_copy(p, idx, sfa, sfb, VEC)
                a0, b0 = address(p.a_bufs.index(2 * idx)), address(p.b_bufs.index(2 * idx))
                a1, b1 = address(p.a_bufs.index(2 * idx + 1)), address(p.b_bufs.index(2 * idx + 1))
                raw_group((a0, a0, a1, a1), (b0, b0, b1, b1),
                          (a0, a0, a1, a1), (b0, b0, b1, b1), acc, sfa, sfb, use_acc,
                          address(p.load_empty_bars.index(idx)), (0, 48, 0, 48), (0, 96 // VEC, 192 // VEC, 288 // VEC),
                          (96, 96, 96, 96), VEC, COMMIT_AFTER=3)
                load_state = load_state.next()
                use_acc = True
        else:
            for k in range(0, K, 256):
                idx = load_state.index
                mbarrier.wait(p.load_ready_bars.index(idx), load_state.phase)
                copy_scales(p, idx, sa, sb, 0, VEC)
                a, b = address(p.a_bufs.index(idx)), address(p.b_bufs.index(idx))
                raw_group((a, a, a), (b, b, b), (a, a, a), (b, b, b), acc, sfa, sfb, use_acc,
                          address(p.load_empty_bars.index(idx)), (0, 48, 96), (0, 96 // VEC, 192 // VEC), (96, 96, 64),
                          VEC, COMMIT_AFTER=2)
                load_state = load_state.next()
                use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def native_mma_partition(p, MODE: gl.constexpr):
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
def mma_partition(p, MODE: gl.constexpr):
    if MODE == "native":
        native_mma_partition(p, MODE)
    else:
        raw_mma_partition(p, MODE)


@gluon.jit
def raw_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE, num_buffers: gl.constexpr,
               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
               num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr, GRID_TILE_WIDTH: gl.constexpr,
               CGA_LAYOUT: gl.constexpr, scheduler: gl.constexpr, MODE: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype,
                                       [num_buffers * (2 if MODE == "exact384" else 1)] + a_desc.block_shape,
                                       a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype,
                                       [num_buffers * (2 if MODE == "exact384" else 1)] + b_desc.block_shape,
                                       b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(
        a_scale_desc.dtype, [num_buffers // 2 if MODE == "exact192" else num_buffers] + a_scale_desc.block_shape,
        a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(
        b_scale_desc.dtype, [num_buffers // 2 if MODE == "exact192" else num_buffers] + b_scale_desc.block_shape,
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
            (mma_partition, (p, MODE)),
            (experimental_load_partition, (p, MODE)),
            (mma_scaled_clc_partition, (p, )),
        ], [1, 1, 1], [24, 24, 24])
    else:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (mma_partition, (p, MODE)),
            (experimental_load_partition, (p, MODE)),
        ], [1, 1], [24, 24])


# The compiler-visible allocation remains power-of-two. Only this experiment's
# marked descriptors override the driver's actual tensor-map box. Barrier byte
# counts and the inline MMA stream below use the exact payload, not that envelope.

_encode_descriptor = _driver.make_tensordesc_arg


def _encode_exact_transfer(arg, metadata, extra):
    if hasattr(arg, '_experimental_transfer_shape'):
        metadata = dict(metadata, block_size=arg._experimental_transfer_shape)
    return _encode_descriptor(arg, metadata, extra)


_driver.make_tensordesc_arg = _encode_exact_transfer


def matmul(A, B, A_scale, B_scale, vec, *, mode='pure', buffers=6, epilogue=32, scheduler=None,
           out_dtype=torch.float16):
    """Return (output, selected binary) for packed FP4 on sm103.

    Scales use example 04's packed block layout. `pure` consumes K768 using
    eight K96 instructions and three packed K256 producer slots. `exact192`
    and `exact384` use exact 96-byte operand TMA boxes, with scales transferred
    once per K384. `raw` is the manually emitted 96+96+64 alignment control.
    This is an inline-PTX experiment, not a general non-power-of-two tensor API.
    """
    assert torch.cuda.get_device_capability(A.device) == (10, 3)
    assert A.dtype == B.dtype == torch.uint8
    assert vec in (16, 32)
    assert mode in ('raw', 'pure', 'native', 'exact192', 'exact384')
    M, N, K = A.shape[0], B.shape[0], A.shape[1] * 2
    assert B.shape[1] * 2 == K
    assert K % (768 if mode in ('pure', 'native') else (384 if mode.startswith('exact') else 256)) == 0
    if mode == 'exact192':
        assert buffers % 2 == 0
    if scheduler is None:
        scheduler = SCHEDULER_SPS if K <= 8192 else SCHEDULER_CLC
    ad, bd, cd, asd, bsd = base.make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N)
    base.mma_scaled_tma_set_block_size_hook(
        dict(a_desc=ad, b_desc=bd, c_desc=cd, a_scale_desc=asd, b_scale_desc=bsd, BLOCK_M=256, BLOCK_N=256,
             BLOCK_K=512 if mode.startswith('exact') else 256, EPILOGUE_BLOCK_N=epilogue, CGA_LAYOUT=((1, 0), )))
    if mode.startswith('exact'):
        ad.block_shape = bd.block_shape = [256, 128]
        ad._experimental_transfer_shape = bd._experimental_transfer_shape = [128, 96]
        asd._experimental_transfer_shape = [1, 1, 384 // (vec * 4), 2, 256]
        bsd._experimental_transfer_shape = [1, 2, 384 // (vec * 4), 2, 256]
    grid = base.mma_scaled_warp_specialized_grid(M, N, 256, 256, 2, scheduler, A.device)
    compiled = raw_kernel[grid](ad, bd, cd, asd, bsd, M, N, K, 2, buffers, 256, 256, 256, epilogue, 1, 0, 8, ((1, 0), ),
                                scheduler, mode, num_ctas=2)
    return cd.base, compiled
