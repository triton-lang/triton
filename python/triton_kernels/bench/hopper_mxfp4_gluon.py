"""Experimental Hopper matmul with MXFP4 operands decoded to E4M3 FP8.

Packing helpers accept E2M1 codes and E8M0 scale bytes in [122, 133].
Values in this range are exactly representable in FP8. This is not a
replacement for the general MXFP4 matmul implementation.
"""
from dataclasses import dataclass
from enum import IntEnum

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, warpgroup_mma


class LhsFormat(IntEnum):
    FP8 = 0
    CANONICAL = 1
    BLOCKED = 2
    WORD_CONVERT = 3
    WORD = 4


class RhsFormat(IntEnum):
    FP8 = 0
    CANONICAL = 1
    SWIZZLED = 2


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = 32
    split_k: int = 4
    row_lanes: int = 16
    row_warps: int = 1

    def __post_init__(self):
        assert self.block_m in (16, 32, 64)
        assert self.split_k in (1, 4, 16, 64)
        assert self.row_lanes in (8, 16, 32)
        assert self.row_warps in (1, 4)


def _check_input(codes, scales):
    if codes.ndim != 2 or codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("Expected a uint8 code matrix and uint8 E8M0 scales")
    rows, k = codes.shape
    if min(rows, k) <= 0 or k % 32 or scales.shape != (rows, k // 32):
        raise ValueError("Expected one scale per 32 K values")
    if codes.device != scales.device:
        raise ValueError("Codes and scales must be on the same device")
    if not bool((codes < 16).all()):
        raise ValueError("E2M1 codes must be in [0, 15]")
    if not bool(((scales >= 122) & (scales <= 133)).all()):
        raise ValueError("The FP8 fast decoder requires scale bytes in [122, 133]")


@g.jit
def decode(q, s):
    # Decode four nibbles per integer instruction; mask zero before restoring its sign.
    bias = ((s.to(gl.uint32) - 121) * 8) * 0x01010101
    q = gl.join(gl.join(q, q), gl.join(q, q)).reshape((q.shape[0], q.shape[1] * 4)).to(gl.uint32)
    bias = gl.join(gl.join(bias, bias), gl.join(bias, bias)).reshape(q.shape)
    return gl.inline_asm_elementwise(
        """{
        .reg .b32 index, value, nonzero, sign;
        and.b32 index, $1, 0x7777;
        prmt.b32 value, 0x0c080000, 0x1c181410, index;
        prmt.b32 nonzero, 0xffffff00, 0xffffffff, index;
        prmt.b32 sign, 0x01010101, 0x01010101, $1;
        xor.b32 sign, sign, 0x01010101;
        shl.b32 sign, sign, 7;
        add.u32 value, value, $5;
        and.b32 value, value, nonzero;
        or.b32 $0, value, sign;
        }""", constraints='=r,r,r,r,r,r,r,r,r', args=[q, bias], dtype=gl.float8e4nv, is_pure=True, pack=4)


@g.jit
def load_weight(W, S, nb, kb, N: gl.constexpr, K: gl.constexpr, WN: gl.constexpr, WK: gl.constexpr, SN: gl.constexpr,
                SK: gl.constexpr, SWIZZLED: gl.constexpr):
    QL: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[8, 0], [0, 4], [0, 8], [0,
                                                                                      16]], lane_bases=[[0, 1], [0, 2],
                                                                                                        [1, 0], [2, 0],
                                                                                                        [4, 0]],
                                                  warp_bases=[[16, 0], [32, 0]], block_bases=[], shape=[64, 32])
    n = gl.arange(0, 64, layout=gl.SliceLayout(1, QL))
    k4 = gl.arange(0, 32, layout=gl.SliceLayout(0, QL))
    if SWIZZLED:
        TL: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]], lane_bases=[[1, 0],
                                                                                                              [2, 0],
                                                                                                              [4, 0],
                                                                                                              [8, 0],
                                                                                                              [16, 0]],
                                                      warp_bases=[[32, 0], [64, 0]], block_bases=[], shape=[128, 16])
        t = gl.arange(0, 128, layout=gl.SliceLayout(1, TL))
        r = gl.arange(0, 16, layout=gl.SliceLayout(0, TL))
        tile = nb * gl.cdiv(K, 128) + kb
        off = (r[None, :] // 8) * 1024 + t[:, None] * 8 + r[None, :] % 8
        q = gl.load(W + tile * 2048 + off, kb < gl.cdiv(K, 128), 0)
        q = q.reshape((4, 8, 4, 4, 2, 2)).trans(0, 5, 1, 3, 4, 2).reshape((64, 32))
        q = gl.convert_layout(q, QL, assert_trivial=True)
        scales = gl.load(S + tile * 64 + n, kb < gl.cdiv(K, 128), 0x7f7f7f7f)
        s = (scales[:, None] >> (k4[None, :] // 8 * 8)) & 255
    else:
        ni = nb * 64 + n
        ki = kb * 32 + k4
        q = gl.load(W + ni[:, None] * WN + ki[None, :] * WK, (ni[:, None] < N) & (ki[None, :] < K // 4), 0)
        s = gl.load(S + ni[:, None] * SN + ki[None, :] // 8 * SK, (ni[:, None] < N) & (ki[None, :] < K // 4), 127)
    return decode(q, s)


@g.jit
def load_lhs(A, S, mb, kb, M: gl.constexpr, K: gl.constexpr, AM: gl.constexpr, AK: gl.constexpr, SM: gl.constexpr,
             SK: gl.constexpr, BM: gl.constexpr, TILED: gl.constexpr, TM: gl.constexpr = 16, TK: gl.constexpr = 128):
    L: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    m = mb * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    j = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
    k4 = kb * 32 + j
    if TILED:
        tile = m[:, None] // TM * gl.cdiv(K, TK) + k4[None, :] // (TK // 4)
        off = tile * (TM * TK // 4) + m[:, None] % TM * (TK // 4) + k4[None, :] % (TK // 4)
        so = tile * (TM * TK // 32) + m[:, None] % TM * (TK // 32) + (k4[None, :] % (TK // 4)) // 8
    else:
        off = m[:, None] * AM + k4[None, :] * AK
        so = m[:, None] * SM + k4[None, :] // 8 * SK
    mask = (m[:, None] < M) & (k4[None, :] < K // 4)
    q = gl.load(A + off, mask, 0)
    scales = gl.load(S + so, mask, 127)
    return decode(q, scales)


@g.jit
def load_lhs_words(A, S, mb, kb, M: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, TM: gl.constexpr = 32,
                   TK: gl.constexpr = 32, LR: gl.constexpr = 32, WR: gl.constexpr = 1):
    # Contiguous lanes traverse rows within each packed-word plane.
    L: gl.constexpr = gl.BlockedLayout([1, 1], [LR, 32 // LR], [WR, 4 // WR], [0, 1])
    m = mb * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    k8 = kb * 16 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    tile = m[:, None] // TM * gl.cdiv(K, TK) + k8[None, :] // (TK // 8)
    off = tile * (TM * TK // 8) + k8[None, :] % (TK // 8) * TM + m[:, None] % TM
    so = tile * (TM * TK // 32) + (k8[None, :] % (TK // 8)) // 4 * TM + m[:, None] % TM
    mask = (m[:, None] < M) & (k8[None, :] < K // 8)
    words = gl.load(A + off, mask, 0)
    scales = gl.load(S + so, mask, 127)
    q = gl.join(words & 65535, words >> 16).reshape((BM, 32))
    scales = gl.join(scales, scales).reshape((BM, 32))
    return decode(q, scales)


@g.jit
def two_sided_kernel(A, AS, W, S, O, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, AM: gl.constexpr,
                     AK: gl.constexpr, WN: gl.constexpr, WK: gl.constexpr, SN: gl.constexpr, SK: gl.constexpr,
                     BM: gl.constexpr, SPLIT: gl.constexpr, SWIZZLED: gl.constexpr, DENSE: gl.constexpr,
                     ASM: gl.constexpr, ASK: gl.constexpr, LHS_MODE: gl.constexpr, LHS_TM: gl.constexpr = 16,
                     LHS_TK: gl.constexpr = 128, WORD_TM: gl.constexpr = 32, WORD_TK: gl.constexpr = 32,
                     WORD_LR: gl.constexpr = 32, WORD_WR: gl.constexpr = 1):
    XL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    ML: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 16, 32])
    DL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=ML, k_width=4)
    SL: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2, transposed=True)
    m = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, XL))
    kk = gl.arange(0, 128, layout=gl.SliceLayout(0, XL))
    acc = gl.zeros((64, BM), gl.float32, ML)
    smem = gl.allocate_shared_memory(gl.float8e4nv, [128, BM], SL)
    for step in range(gl.cdiv(K, 128 * SPLIT)):
        kb = step * SPLIT + gl.program_id(2)
        if DENSE:
            WL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
            ni = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, WL))
            ki = kb * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, WL))
            w = gl.load(W + ni[:, None] * WN + ki[None, :] * WK, (ni[:, None] < N) & (ki[None, :] < K), 0.0)
            w = gl.convert_layout(w, DL)
        else:
            w = load_weight(W, S, gl.program_id(1), kb, N, K, WN, WK, SN, SK, SWIZZLED)
            w = gl.convert_layout(w, DL, assert_trivial=True)
        ki = kb * 128 + kk
        if LHS_MODE == 0:
            x = gl.load(A + m[:, None] * AM + ki[None, :] * AK, (m[:, None] < M) & (ki[None, :] < K), 0.0)
        elif LHS_MODE >= 3:
            x = load_lhs_words(A, AS, gl.program_id(0), kb, M, K, BM, WORD_TM, WORD_TK, WORD_LR, WORD_WR)
            if LHS_MODE == 3:
                x = gl.convert_layout(x, XL)
        else:
            x = load_lhs(A, AS, gl.program_id(0), kb, M, K, AM, AK, ASM, ASK, BM, LHS_MODE == 2, LHS_TM, LHS_TK)
            x = gl.convert_layout(x, XL)
        smem.store(x.T)
        fence_async_shared()
        # Evaluate the transposed product: logical RHS in registers, LHS in shared memory.
        tile = warpgroup_mma(w, smem, gl.zeros((64, BM), gl.float32, ML), max_num_imprecise_acc=128)
        # Keep FP32 tile additions explicit rather than extending imprecise MMA accumulation.
        acc = gl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[acc, tile],
                                        dtype=gl.float32, is_pure=True, pack=1)
    ni = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, ML))
    mi = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, ML))
    gl.store(O + gl.program_id(2) * M * N + mi[None, :] * N + ni[:, None], acc, (mi[None, :] < M) & (ni[:, None] < N))


def pack_rhs(codes, scales, swizzled):
    _check_input(codes, scales)
    n, k = codes.shape
    if swizzled:
        padded = torch.zeros((triton.cdiv(n, 64) * 64, triton.cdiv(k, 128) * 128), device=codes.device,
                             dtype=torch.uint8)
        padded[:n, :k] = codes
        ss = torch.full((padded.shape[0], padded.shape[1] // 32), 127, device=scales.device, dtype=torch.uint8)
        ss[:n, :k // 32] = scales
        codes, scales = padded, ss
    nn, kk = codes.shape
    q = codes.to(torch.int32).reshape(nn, kk // 4, 4)
    q = (q[..., 0] | q[..., 1] << 4 | q[..., 2] << 8 | q[..., 3] << 12).to(torch.int16)
    if swizzled:
        q = q.reshape(nn // 64, 4, 2, 8, kk // 128, 4, 2, 4).permute(0, 4, 1, 3, 7, 5, 6, 2).contiguous()
        q = q.reshape(nn // 64, kk // 128, 128, 2, 8).permute(0, 1, 3, 2, 4).contiguous()
        scales = scales.reshape(nn // 64, 64, kk // 128, 4).permute(0, 2, 1, 3).contiguous().view(torch.int32)
    return q.view(torch.uint16), scales


def pack_lhs_words(codes, scales, tile=(16, 64)):
    """Store packed-word planes, with contiguous rows in each plane."""
    _check_input(codes, scales)
    m, k = codes.shape
    tm, tk = tile
    assert tm in (8, 16, 32, 64) and tk in (32, 64, 128, 256)
    assert k % 32 == 0
    padded = torch.zeros((triton.cdiv(m, tm) * tm, triton.cdiv(k, tk) * tk), device=codes.device, dtype=torch.uint8)
    padded[:m, :k] = codes
    ss = torch.full((padded.shape[0], padded.shape[1] // 32), 127, device=codes.device, dtype=torch.uint8)
    ss[:m, :k // 32] = scales
    q, _ = pack_rhs(padded, ss, False)
    q = q.view(torch.int32).reshape(-1, tm, padded.shape[1] // tk, tk // 8).permute(0, 2, 3,
                                                                                    1).contiguous().view(torch.uint32)
    ss = ss.reshape(-1, tm, padded.shape[1] // tk, tk // 32).permute(0, 2, 3, 1).contiguous()
    return q, ss


@g.jit
def pack_kernel(W, S, Q, T, N: gl.constexpr, K: gl.constexpr, WN: gl.constexpr, WK: gl.constexpr, SN: gl.constexpr,
                SK: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    i = gl.arange(0, 2048, layout=L)
    t = i % 1024 // 8
    r = i // 1024 * 8 + i % 8
    n = gl.program_id(0) * 64 + t // 32 * 16 + r % 2 * 8 + t % 32 // 4
    k4 = gl.program_id(1) * 32 + r // 2 * 4 + t % 4
    q = gl.load(W + n * WN + k4 * WK, (n < N) & (k4 < K // 4), 0)
    tile = gl.program_id(0) * gl.cdiv(K, 128) + gl.program_id(1)
    gl.store(Q + tile * 2048 + i, q)
    SL: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    j = gl.arange(0, 256, layout=SL)
    ns = gl.program_id(0) * 64 + j // 4
    ks = gl.program_id(1) * 4 + j % 4
    s = gl.load(S + ns * SN + ks * SK, (ns < N) & (ks < K // 32), 127)
    gl.store(T + tile * 256 + j, s)


def make_rhs_packer(packed, scales, k):
    """Prepare tiled storage from canonical four-nibbles-per-word MXFP4."""
    n = scales.shape[0]
    q = torch.empty((triton.cdiv(n, 64), triton.cdiv(k, 128), 2048), device=packed.device, dtype=torch.uint16)
    s = torch.empty((triton.cdiv(n, 64), triton.cdiv(k, 128), 256), device=scales.device, dtype=torch.uint8)

    def run():
        return pack_kernel[(triton.cdiv(n, 64), triton.cdiv(k, 128))](packed, scales, q, s, n, k, *packed.stride(),
                                                                      *scales.stride())

    run()
    return q, s.view(torch.int32), run


def pack_lhs_rows(codes, scales, tile=None):
    _check_input(codes, scales)
    if tile is None:
        return pack_rhs(codes, scales, False)
    m, k = codes.shape
    tm, tk = tile
    assert tm > 0 and tk >= 32 and tk % 32 == 0
    padded = torch.zeros((triton.cdiv(m, tm) * tm, triton.cdiv(k, tk) * tk), device=codes.device, dtype=torch.uint8)
    ss = torch.full((padded.shape[0], padded.shape[1] // 32), 127, device=codes.device, dtype=torch.uint8)
    padded[:m, :k], ss[:m, :k // 32] = codes, scales
    q, _ = pack_rhs(padded, ss, False)
    q = q.view(torch.int16).reshape(-1, tm, padded.shape[1] // tk, tk // 4).permute(0, 2, 1,
                                                                                    3).contiguous().view(torch.uint16)
    ss = ss.reshape(-1, tm, padded.shape[1] // tk, tk // 32).permute(0, 2, 1, 3).contiguous()
    return q, ss


@triton.jit
def reduce_kernel(P, O, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT)
    values = tl.load(P + splits[:, None] * SIZE + offsets[None, :], offsets[None, :] < SIZE, other=0)
    tl.store(O + offsets, tl.sum(values, 0), offsets < SIZE)


@triton.jit
def pack_lhs_kernel(Q, S, W, T, M: tl.constexpr, K: tl.constexpr, TM: tl.constexpr, TK: tl.constexpr):
    i = tl.program_id(0) * 512 + tl.arange(0, 512)
    nt: tl.constexpr = triton.cdiv(K, TK)
    total: tl.constexpr = triton.cdiv(M, TM) * nt * TM * TK // 8
    tile = i // (TM * TK // 8)
    row = tile // nt * TM + i % TM
    col = tile % nt * (TK // 8) + i % (TM * TK // 8) // TM
    q = tl.load(Q + row * (K // 8) + col, (i < total) & (row < M) & (col < K // 8), 0)
    tl.store(W + i, q, i < total)
    st = i // (TM * TK // 32)
    sr = st // nt * TM + i % TM
    sc = st % nt * (TK // 32) + i % (TM * TK // 32) // TM
    s = tl.load(S + sr * (K // 32) + sc, (i < total // 4) & (sr < M) & (sc < K // 32), 127)
    tl.store(T + i, s, i < total // 4)


def make_lhs_packer(packed, scales, tile=(16, 64)):
    """Return output buffers and a reusable canonical-to-word packing operation."""
    m, scale_k = scales.shape
    k = scale_k * 32
    tm, tk = tile
    assert tm in (8, 16, 32, 64) and tk in (32, 64, 128, 256)
    shape = (triton.cdiv(m, tm), triton.cdiv(k, tk))
    q = torch.empty((*shape, tk // 8, tm), device=packed.device, dtype=torch.uint32)
    s = torch.empty((*shape, tk // 32, tm), device=scales.device, dtype=torch.uint8)
    packed = packed.view(torch.uint32)
    assert packed.is_contiguous() and scales.is_contiguous()

    def run():
        pack_lhs_kernel[(triton.cdiv(q.numel(), 512), )](packed, scales, q, s, m, k, tm, tk)

    run()
    return q, s, run


def make_runner(a, sa, w, sw, shape, config=KernelConfig(), *, lhs_format=LhsFormat.WORD, rhs_format=RhsFormat.SWIZZLED,
                lhs_tile=(16, 64)):
    """Build a reusable operation; allocations and packing remain outside timing.

    Shape is logical (M, N, K), with inputs stored as (M, K) and (N, K).
    Raw packed inputs must satisfy the packing helpers' scale contract.
    """
    m, n, k = shape
    assert min(shape) > 0 and k % 32 == 0
    assert isinstance(lhs_format, LhsFormat) and isinstance(rhs_format, RhsFormat)
    if lhs_format in (LhsFormat.WORD, LhsFormat.WORD_CONVERT):
        assert lhs_tile[0] in (8, 16, 32, 64) and lhs_tile[1] in (32, 64, 128, 256)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    split = config.split_k
    partial = out if split == 1 else torch.empty((split, m, n), device=a.device, dtype=torch.float32)
    am, ak = a.stride() if lhs_format in (LhsFormat.FP8, LhsFormat.CANONICAL) else (0, 0)
    asm, ask = sa.stride() if lhs_format == LhsFormat.CANONICAL else (0, 0)
    wn, wk = (0, 0) if rhs_format == RhsFormat.SWIZZLED else w.stride()
    sn, sk = sw.stride() if rhs_format == RhsFormat.CANONICAL else (0, 0)

    def run():
        compiled = two_sided_kernel[(triton.cdiv(m, config.block_m), triton.cdiv(n, 64),
                                     split)](a, sa, w, sw, partial, m, n, k, am, ak, wn, wk, sn, sk, config.block_m,
                                             split,
                                             rhs_format == RhsFormat.SWIZZLED, rhs_format == RhsFormat.FP8, asm, ask,
                                             int(lhs_format), *lhs_tile, *lhs_tile, config.row_lanes, config.row_warps,
                                             num_warps=4, enable_fp_fusion=False)
        if split > 1:
            reduce_kernel[(triton.cdiv(m * n, 256), )](partial, out, m * n, split, 256)
        return compiled

    return run, out
