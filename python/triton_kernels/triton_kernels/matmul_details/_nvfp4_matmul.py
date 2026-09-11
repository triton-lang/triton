"""Persistent dense NVFP4 specialization for the matmul dispatcher."""

import torch
import triton
import triton.language as tl

from triton_kernels.tensor import FP4, make_tma
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    StridedLayout,
)
from triton_kernels.tensor_details.layout_details.blackwell_scale import (
    unswizzle_act_mx_scale_bw,
    unswizzle_mx_scale_bw,
)


@triton.jit
def _nvfp4_matmul(A, B, SA, SB, C, ScaleA, ScaleB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, SMS: tl.constexpr, GROUP: tl.constexpr, EPILOGUE: tl.constexpr):
    for tile in tl.range(tl.program_id(0), M // BM * (N // BN), SMS, flatten=True, warp_specialize=True,
                         disallow_acc_multi_buffer=True):
        group = tile // (GROUP * (N // BN))
        pm = group * GROUP + tile % GROUP
        pn = tile % (GROUP * (N // BN)) // GROUP
        acc = tl.zeros((BM, BN), tl.float32)
        for kk in tl.range(K // BK, disallow_acc_multi_buffer=True):
            a = A.load([pm * BM, kk * BK // 2])
            b = B.load([pn * BN, kk * BK // 2])
            sa = SA.load([0, pm * (BM // 128), kk * (BK // 64), 0, 0])
            sb = SB.load([0, pn * (BN // 128), kk * (BK // 64), 0, 0])
            sa = unswizzle_act_mx_scale_bw(sa)
            sb = sb.reshape((sb.shape[1], sb.shape[2] * sb.shape[-2] * sb.shape[-1]))
            sb = unswizzle_mx_scale_bw(sb)
            acc = tl.dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc, fast_math=True)
        scale = 1.0
        if ScaleA is not None:
            scale *= tl.load(ScaleA)
        if ScaleB is not None:
            scale *= tl.load(ScaleB)
        acc *= scale
        rows = pm * BM + tl.arange(0, BM)
        if M * N >= 2**31:
            rows = rows.to(tl.int64)
        if EPILOGUE == 1:
            parts = (acc, )
        else:
            left, right = acc.reshape(BM, 2, BN // 2).permute(0, 2, 1).split()
            if EPILOGUE == 2:
                parts = (left, right)
            else:
                a0, a1 = left.reshape(BM, 2, BN // 4).permute(0, 2, 1).split()
                a2, a3 = right.reshape(BM, 2, BN // 4).permute(0, 2, 1).split()
                parts = (a0, a1, a2, a3)
        for j in tl.static_range(EPILOGUE):
            cols = pn * BN + j * (BN // EPILOGUE) + tl.arange(0, BN // EPILOGUE)
            tl.store(C + rows[:, None] * N + cols[None, :], parts[j])


def can_use_nvfp4_matmul(a, b, out, a_scale, b_scale, precision, flags):
    if not flags.is_persistent or flags.clc or flags.split_k != 1:
        return False
    if flags.swap_xw is True or flags.use_output_tma is True:
        return False
    if not flags.target_kernel_kwargs.get("FLATTEN_LOOPS", False):
        return False
    if flags.block_m != 128 or flags.block_n not in (128, 256) or flags.block_k % 128:
        return False
    if flags.num_warps not in (4, 8) or flags.epilogue_subtile not in (1, 2, 4):
        return False
    if a.ndim != 2 or b.ndim != 2 or a.storage.data.ndim != 2 or b.storage.data.ndim != 2:
        return False
    if a.dtype != FP4 or b.dtype != FP4 or out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    if precision.enforce_bitwise_invariance or precision.c_mx_scale is not None:
        return False
    if precision.a_microblock_size != 16 or precision.b_microblock_size != 16:
        return False
    if type(a.storage.layout) is not StridedLayout or type(b.storage.layout) is not BlackwellMXValueLayout:
        return False
    if a_scale is None or b_scale is None:
        return False
    if type(a_scale.storage.layout) is not BlackwellActMXScaleLayout:
        return False
    if a_scale.storage.layout.ragged_metadata is not None:
        return False
    if type(b_scale.storage.layout) is not BlackwellMXScaleLayout:
        return False
    if a_scale.storage.data.dtype != torch.float8_e4m3fn or b_scale.storage.data.dtype != torch.float8_e4m3fn:
        return False
    for scale in (precision.a_mx_tensor_scale, precision.b_mx_tensor_scale):
        if scale is not None and (scale.numel() != 1 or scale.dtype != torch.float32):
            return False
    m, k = a.shape
    n = b.shape[1]
    return m % (flags.block_m * flags.group_m) == 0 and n % flags.block_n == 0 and k % flags.block_k == 0


def launch_nvfp4_matmul(a, b, out, a_scale, b_scale, a_tensor_scale, b_tensor_scale, flags, grid):
    ad = make_tma(a, [flags.block_m, flags.block_k], "dense")
    bd = make_tma(b, [flags.block_k, flags.block_n], "dense")
    sad = make_tma(a_scale, [flags.block_m, flags.block_k // 16], "dense", is_scale=True)
    sbd = make_tma(b_scale, [flags.block_k // 16, flags.block_n], "dense", is_scale=True)
    m, k = a.shape
    n = b.shape[1]
    return _nvfp4_matmul[(grid, )](
        ad,
        bd,
        sad,
        sbd,
        out,
        a_tensor_scale,
        b_tensor_scale,
        m,
        n,
        k,
        flags.block_m,
        flags.block_n,
        flags.block_k,
        grid,
        flags.group_m,
        flags.epilogue_subtile,
        num_warps=flags.num_warps,
        num_stages=flags.num_stages,
        arch=flags.arch,
        maxnreg=flags.target_kernel_kwargs.get("maxnreg"),
    )
