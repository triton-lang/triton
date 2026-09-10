"""Self-contained Gluon/inline-PTX dense MXFP4 GEMM for SM103.

    python examples/dense_k96.py --m 16384 --n 16384 --k 16128

BLOCK_K=768 consists of eight native K96 MMAs, with no K64 tail.
A and B are row-major packed E2M1 (B is logically transposed). Scales are
per-32 E8M0, packed in the documented tcgen05 scale layout. Output is FP16 or BF16.
The benchmark measures GEMM only; scale packing and descriptor creation are
outside the timed region. Use --sweep for full correctness and three sizes.

The schedule follows 07-pure-k96-matmul.py at historical commit 9961ab025fa8:
six K256 producer slots, early slot releases, and a persistent planar snake.
Descriptor operands use public Gluon inline assembly; no compiler changes or
post-compilation patches are needed. Allocation, TMA, scale copies, barriers,
issuer selection, and the epilogue use native Gluon.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory,
    mbarrier, tma, tcgen05_copy, tcgen05_commit,
)
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 768
NUM_BUFFERS = 6


@builtin
def two_cta_mma(_semantic=None, _generator=None):
    # Inline MMA is opaque to mode inference. Declare its mode while building
    # the initial module so native TMA/copy/commit use the same CTA pair.
    _generator.module.set_attr("ttng.two-ctas", _semantic.builder.get_bool_attr(True))


@g.jit
def shared_descriptor(desc):
    start = gl.inline_asm("shr.b32 $0, $1, 4;", "=r,r", [desc], gl.uint32)
    return (start & 16383).to(gl.uint64) | (1 << 62) | (1 << 46) | (64 << 32)


@g.jit
def mma_k96(a_bufs, b_bufs, acc, a_scales, b_scales, slot, k: gl.constexpr, use_acc):
    a = shared_descriptor(a_bufs.index(slot + k // 256)) + k % 256 // 32
    b = shared_descriptor(b_bufs.index(slot + k // 256)) + k % 256 // 32
    if k % 256 + 96 > 256:
        next_a = shared_descriptor(a_bufs.index(slot + k // 256 + 1))
        next_b = shared_descriptor(b_bufs.index(slot + k // 256 + 1))
        a |= ((next_a & 16383) << 16) | (1 << 52)
        b |= ((next_b & 16383) << 16) | (1 << 52)
    sid: gl.constexpr = (k // 32) % 4
    # SM103 selects native K96 with instruction-descriptor bit 31.
    descriptor: gl.constexpr = 0x90c00480 | (sid << 29) | (sid << 4)
    # A crossing MMA reads scale values from both four-scale groups.
    scale_count: gl.constexpr = 8 if sid > 1 else 4
    a_scale = a_scales.slice(k // 128 * 4, scale_count)
    b_scale = b_scales.slice(k // 128 * 4, scale_count)
    # This layout spans the one-warp consumer in both CTAs. Predicates remain
    # tensors, with one value per participating thread.
    owner = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [1], [0], cga_layout=[[1]]))
    # Retain the explicit CTA branch: predicating only MMA costs about 7%.
    gl.inline_asm("""{
@$6 bra.uni END;
tcgen05.fence::after_thread_sync;
@$7 tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32
    [$0], $1, $2, $8, [$3], [$4], $5;
END:
}""", "r,l,l,r,r,b,b,b,r", [acc, a, b, a_scale, b_scale, use_acc, owner >= 32, owner == 0, descriptor])


@g.jit
def tile_coords(tile, M: gl.constexpr, N: gl.constexpr, width: gl.constexpr):
    group = tile // (width * (N // 256))
    first = group * width
    size = gl.minimum(M // 256 - first, width)
    m = (first + tile % size) * 256
    n = (tile % (width * (N // 256)) // size) * 256
    return m, gl.where(group % 2 == 0, n, N - 256 - n)


@g.jit
def load_partition(A, B, SA, SB, a_bufs, b_bufs, sa_bufs, sb_bufs, ready, empty,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    ordinal = 0
    for tile in range(gl.program_id(0), M // 256 * (N // 256), gl.num_programs(0)):
        m, n = tile_coords(tile, M, N, 16 if K <= 16384 else 8)
        for i in range(K // 256):
            sequence = ordinal * (K // 256) + i
            slot = sequence % 6
            a, b = a_bufs.index(slot), b_bufs.index(slot)
            sa, sb = sa_bufs.index(slot), sb_bufs.index(slot)
            mbarrier.wait(empty.index(slot), (sequence // 6 % 2) ^ 1, deps=(a, b, sa, sb))
            bar = ready.index(slot)
            mbarrier.expect(bar, A.nbytes_per_cta + B.nbytes_per_cta + SA.nbytes_per_cta + SB.nbytes_per_cta)
            tma.async_load(A, [m, i * 128], bar, a)
            tma.async_load(B, [n, i * 128], bar, b)
            tma.async_load(SA, [m // 128, i * 2, 0, 0], bar, sa)
            tma.async_load(SB, [n // 128, i * 2, 0, 0], bar, sb)
        ordinal += 1


@g.jit
def unpack_scale_tile(smem):
    return smem.reshape((2, 2, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((256, 8))


@g.jit
def mma_partition(a_bufs, b_bufs, sa_bufs, sb_bufs, ready, empty, acc, acc_ready, acc_empty,
                  M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    a_scales = allocate_tensor_memory(gl.uint8, [256, 32], TensorMemoryScalesLayout([[1, 0]]))
    b_scales = allocate_tensor_memory(gl.uint8, [256, 32], TensorMemoryScalesLayout([[0, 0]]))
    ordinal = 0
    for tile in range(gl.program_id(0), M // 256 * (N // 256), gl.num_programs(0)):
        mbarrier.wait(acc_empty, (ordinal % 2) ^ 1, deps=(acc,))
        for i in range(K // 768):
            sequence = (ordinal * (K // 768) + i) * 3
            slot = sequence % 6
            for sector in gl.static_range(3):
                sa = sa_bufs.index(slot + sector)
                sb = sb_bufs.index(slot + sector)
                mbarrier.wait(ready.index(slot + sector), sequence // 6 % 2,
                              deps=(a_bufs.index(slot + sector), b_bufs.index(slot + sector), sa, sb))
                tcgen05_copy(unpack_scale_tile(sa), a_scales.slice(sector * 8, 8))
                tcgen05_copy(unpack_scale_tile(sb), b_scales.slice(sector * 8, 8))
                for k in gl.static_range((0, 192, 480)[sector], (192, 480, 768)[sector], 96):
                    mma_k96(a_bufs, b_bufs, acc, a_scales, b_scales, slot, k, (i > 0) | (k > 0))
                    if k == 192 or k == 480 or k == 672:
                        released = 0 if k == 192 else (1 if k == 480 else 2)
                        tcgen05_commit(empty.index(slot + released),
                                       descs=(a_bufs.index(slot + released), b_bufs.index(slot + released),
                                              sa_bufs.index(slot + released), sb_bufs.index(slot + released)))
        tcgen05_commit(acc_ready)
        ordinal += 1


@g.jit
def epilogue_partition(acc, acc_ready, acc_empty, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    source: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
        [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], [[32, 0], [64, 0], [16, 0]], [[128, 0]], [256, 256])
    target: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [0, 64], [0, 128]],
        [[1, 0], [2, 0], [0, 8], [0, 16], [0, 32]], [[32, 0], [64, 0], [16, 0]], [[128, 0]], [256, 256])
    rr = gl.arange(0, 256, layout=gl.SliceLayout(1, target))
    cc = gl.arange(0, 256, layout=gl.SliceLayout(0, target))
    ordinal = 0
    for tile in range(gl.program_id(0), M // 256 * (N // 256), gl.num_programs(0)):
        m, n = tile_coords(tile, M, N, 16 if K <= 16384 else 8)
        mbarrier.wait(acc_ready, ordinal % 2, deps=(acc,))
        values = acc.load(source)
        mbarrier.arrive(acc_empty)
        values = values.to(C.dtype.element_ty)
        gl.store(C + (m + rr[:, None]) * N + n + cc[None, :], gl.convert_layout(values, target))
        ordinal += 1


@g.jit
def dense_k96(A, B, SA, SB, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    two_cta_mma()
    a = gl.allocate_shared_memory(gl.uint8, [6] + A.block_shape, A.layout)
    b = gl.allocate_shared_memory(gl.uint8, [6] + B.block_shape, B.layout)
    sa = gl.allocate_shared_memory(gl.uint8, [6] + SA.block_shape, SA.layout)
    sb = gl.allocate_shared_memory(gl.uint8, [6] + SB.block_shape, SB.layout)
    ready = mbarrier.allocate_mbarrier(batch=6, two_ctas=True)
    empty = mbarrier.allocate_mbarrier(batch=6)
    acc_ready = mbarrier.allocate_mbarrier()
    acc_empty = mbarrier.allocate_mbarrier(two_ctas=True)
    for i in gl.static_range(6):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    mbarrier.init(acc_ready, count=1)
    mbarrier.init(acc_empty, count=1)
    acc = allocate_tensor_memory(gl.float32, [256, 256],
                                 TensorMemoryLayout([128, 256], col_stride=1, cga_layout=[[1, 0]], two_ctas=True))
    gl.warp_specialize([
        (epilogue_partition, (acc, acc_ready, acc_empty, C, M, N, K)),
        (mma_partition, (a, b, sa, sb, ready, empty, acc, acc_ready, acc_empty, M, N, K)),
        (load_partition, (A, B, SA, SB, a, b, sa, sb, ready, empty, M, N, K)),
    ], [1, 1], [32, 32])
    gl.barrier(cluster=True)
    for i in gl.static_range(6):
        mbarrier.invalidate(ready.index(i))
        mbarrier.invalidate(empty.index(i))
    mbarrier.invalidate(acc_ready)
    mbarrier.invalidate(acc_empty)


def prepare(rows, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    data = torch.randint(0, 256, (rows, k // 2), generator=generator, device="cuda", dtype=torch.uint8)
    scales = torch.randint(126, 129, (rows, k // 32), generator=generator, device="cuda", dtype=torch.uint8)
    return data, scales


def pack_scales(scales):
    rows, cols = scales.shape
    return scales.reshape(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).contiguous()


def decode(data, scales):
    # Independent oracle: decode the original row-major scales, not the kernel's
    # packed scale buffer or its shared/TMEM layout.
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=data.device)
    values = table[torch.stack((data & 15, data >> 4), dim=-1).long()].flatten(1)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)


def make_runner(a, b, a_scales, b_scales, out_dtype=torch.float16):
    """Prepare a reusable dense A @ B.T launch; packing is outside GPU timing.

    Inputs are CUDA uint8 tensors: A[M,K/2], B[N,K/2], and E8M0 scales
    [M,K/32], [N,K/32]. M,N must be multiples of 256; K a multiple of 768.
    out_dtype is torch.float16 or torch.bfloat16.
    """
    m, k = a.shape[0], a.shape[1] * 2
    n = b.shape[0]
    assert m > 0 and n > 0 and k > 0
    assert m % BLOCK_M == 0 and n % BLOCK_N == 0 and k % BLOCK_K == 0
    packed_a, packed_b = pack_scales(a_scales), pack_scales(b_scales)
    data_layout = gl.NVMMASharedLayout(128, 8, cga_layout=[[1, 0]])
    a_map = TensorDescriptor.from_tensor(a, [256, 128], data_layout)
    b_map = TensorDescriptor.from_tensor(b, [256, 128], data_layout)
    sa_map = TensorDescriptor.from_tensor(packed_a.reshape(m // 128, k // 128, 4, 128), [2, 2, 4, 128],
                                          gl.NVMMASharedLayout(0, 8, rank=4, cga_layout=[[1, 0, 0, 0]]))
    sb_map = TensorDescriptor.from_tensor(packed_b.reshape(n // 128, k // 128, 4, 128), [2, 2, 4, 128],
                                          gl.NVMMASharedLayout(0, 8, rank=4, cga_layout=[[0, 0, 0, 0]]))
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    clusters = min(torch.cuda.get_device_properties(a.device).multi_processor_count // 2,
                   m // BLOCK_M * (n // BLOCK_N))

    def launch():
        return dense_k96[(clusters,)](a_map, b_map, sa_map, sb_map, out, m, n, k, num_ctas=2, num_warps=8)

    return out, launch


def run(m, n, k, *, out_dtype=torch.float16, check="full", bench=True, dump=None):
    a, a_scales = prepare(m, k, 17)
    b, b_scales = prepare(n, k, 29)
    out, launch = make_runner(a, b, a_scales, b_scales, out_dtype)
    kernel = launch()
    torch.cuda.synchronize()
    if check == "full":
        reference = decode(a, a_scales) @ decode(b, b_scales).T
        actual = out
    else:
        rows = torch.arange(0, m, max(1, m // 64), device=a.device)
        cols = torch.arange(0, n, max(1, n // 64), device=a.device)
        reference = decode(a[rows], a_scales[rows]) @ decode(b[cols], b_scales[cols]).T
        actual = out[rows[:, None], cols]
    torch.testing.assert_close(actual, reference.to(out.dtype), atol=0, rtol=0)
    checked = reference.numel()
    print(f"Correct: {checked} {check} outputs, exact output comparison", flush=True)
    del reference, actual

    result = {"m": m, "n": n, "k": k, "check": check, "checked_outputs": checked,
              "out_dtype": str(out_dtype), "format": "mxfp4",
              "block_k": BLOCK_K, "gpu": torch.cuda.get_device_name(a.device),
              "triton": triton.__version__, "triton_path": triton.__file__}
    if bench:
        samples = [triton.testing.do_bench_cudagraph(launch, rep=100) for _ in range(5)]
        ms = statistics.median(samples)
        result.update(ms=ms, dense_pflops=2 * m * n * k / ms / 1e12, samples_ms=samples)
        print(f"{m}x{n}x{k}: {ms:.6f} ms, {result['dense_pflops']:.4f} dense PFLOP/s; {samples}", flush=True)
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        for ext in ("ptx", "ttgir", "cubin"):
            path = dump / f"dense_k96.{ext}"
            if ext == "cubin":
                path.write_bytes(kernel.asm[ext])
            else:
                path.write_text(kernel.asm[ext])
        (dump / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--k", type=int, default=16128)
    parser.add_argument("--out-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--check", choices=("full", "sample"), default="full")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Native K96 requires an SM103 GPU (GB300/B300)")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(torch.cuda.get_device_name(), triton.__file__, flush=True)
    shapes = [(8192, 8192, 7680), (16384, 16384, 16128), (32768, 32768, 32256)] if args.sweep else [(args.m, args.n, args.k)]
    for m, n, k in shapes:
        dump = args.dump / f"{m}x{n}x{k}" if args.dump else None
        run(m, n, k, out_dtype=getattr(torch, args.out_dtype), check=args.check, bench=not args.check_only, dump=dump)


if __name__ == "__main__":
    main()
