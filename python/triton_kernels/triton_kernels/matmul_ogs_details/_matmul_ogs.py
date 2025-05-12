import triton
import triton.language as tl
from triton_kernels.numerics_details.mxfp import _unswizzle_mx_block, get_scaled_dot_format_string
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from ._common import make_matmul_repr, matmul_launch_metadata, swizzle2d, xcd_swizzle

# fmt: off

@triton.jit
def _zero_masked_rows(
        pid_m, pid_n,
        Y, stride_y_m, stride_y_n,
        N,
        ScatterSrcIndx, num_idxs,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = BLOCK_M * pid_m.to(tl.int64) + tl.arange(0, BLOCK_M)
    offs_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    src_idx = tl.load(ScatterSrcIndx + offs_m, mask=offs_m < num_idxs, other=0)
    YPtrs = Y + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    mask_n = offs_n < N
    mask = (src_idx == -1)[:, None] & mask_n[None, :]
    tl.store(YPtrs, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), mask=mask)


_matmul_ogs_repr = make_matmul_repr("_matmul_ogs", [0, 1, 2])
@triton.jit(repr=_matmul_ogs_repr, launch_metadata=matmul_launch_metadata)
def _matmul_ogs(
             Y, Out, stride_y_k, stride_y_z, stride_y_m, stride_y_n,
             YExpectedScale, YActualScale, YChecksumScale,
             X, stride_x_z, stride_x_m, stride_x_k,
             XScale,
             W, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: tl.constexpr,
             WScale,
             MxScale, stride_mx_e, stride_mx_k, stride_mx_n, MX_TRANSPOSE: tl.constexpr,
             B, stride_b_e, # Bias
             NRows, M, N, K, # shapes
             # expt data
             Betas, Gammas,
             GatherIndx,
             ScatterSrcIndx, num_idxs,
             WriteBackIndx, writeback_size,
             ExptHist, ExptOffs, ExptOffsSum, ExptData,
             # true grid size
             batch_size, grid_m, grid_n,
             # Out scale
             out_alpha,
             # epilogue transform
             EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
             # MoE config
             N_EXPTS_TOT: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
             # precision config
             MAX_NUM_IMPRECISE_ACC: tl.constexpr, ALLOW_TF32: tl.constexpr,
             FLEXPOINT_SATURATE_INF: tl.constexpr,
             PER_BATCH_SCALE: tl.constexpr,
             # optimization config
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr, SWIZZLE_MX: tl.constexpr,
             EPILOGUE_SUBTILE: tl.constexpr,
             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
             W_CACHE_MODIFIER: tl.constexpr,
             NUM_SMS: tl.constexpr,
             TOKENS_PER_EXPT_FOR_ANNOTATION=None,
             UPCAST_INDICES: tl.constexpr = False,
             DISABLE_Y_TMA: tl.constexpr = True,
             SWAP_XW: tl.constexpr = False):

    Y = Out  # Y is passed for the purposes of annotation; replace it with Out
    is_microscaled_format: tl.constexpr = MxScale is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    if is_microscaled_format:
        w_type: tl.constexpr = W.dtype.element_ty
        tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                         "mx_weight_ptr must be uint8")
        tl.static_assert(MxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    HAS_FUSED_SCATTER: tl.constexpr = WriteBackIndx is not None
    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    total_actual_tiles = batch_size * (grid_m - padding_m) * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        tl.device_assert(batch_size == 0)
        pid_mn = pid - total_actual_tiles
        if pid_mn < padding_m * grid_n:
            pid_m, pid_n = swizzle2d(pid_mn, padding_m, grid_n, GROUP_M)

            # set masked out rows to 0
            if HAS_FUSED_SCATTER and N_EXPTS_ACT == 1:
                _zero_masked_rows(pid_m, pid_n, Y, stride_y_m, stride_y_n, N, ScatterSrcIndx, num_idxs, BLOCK_M, BLOCK_N)
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_e = pid_emnk // ((grid_m - padding_m) * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % ((grid_m - padding_m) * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = swizzle2d(pid_mn, (grid_m - padding_m), grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to( index_type) * stride_y_k
    # set masked out rows to 0
    if HAS_FUSED_SCATTER and N_EXPTS_ACT == 1:
        _zero_masked_rows(pid_m, pid_n, Y, stride_y_m, stride_y_n, N, ScatterSrcIndx, num_idxs, BLOCK_M, BLOCK_N)
    # unpack expert data
    if ExptData is None:
        tl.static_assert(M is not None)
        expt_id, start_z, start_m, block_id = pid_e, pid_e, 0, pid_m
    else:
        tl.static_assert(M is None)
        expt_data = tl.load(ExptData + pid_m)
        if expt_data == -1:
            return
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        M = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        start_z = 0
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m, start_z = start_m.to(index_type), start_z.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)
    # A pointers
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    X += start_z * stride_x_z
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m.to(index_type)[:, None] * stride_x_m + offs_k.to(index_type)[None, :] * stride_x_k
    # B pointers
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)

    # TODO: refactor if/else when triton front end improves
    if is_microscaled_format:
        # We have pack 2 fp4 values in a  byte
        W_PACK_DIVISOR: tl.constexpr = 2 if W.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_PACK_DIVISOR
        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        MxScale += expt_id * stride_mx_e

        if SWIZZLE_MX:
            tl.static_assert(BLOCK_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K % 4 == 0)
            PACKED_MX_BLOCK: tl.constexpr = (MX_SCALE_BLOCK_K // 4) * 32 * 4 * 4
            offs_inner = PACKED_MX_BLOCK * pid_k + tl.arange(0, PACKED_MX_BLOCK)
            offs_n_scale = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N
            offs_n_scale = tl.max_contiguous(tl.multiple_of(offs_n_scale, BLOCK_N // 128), BLOCK_N // 128)

            MxScalePtrs = MxScale + offs_n_scale.to(index_type)[:, None] * stride_mx_n + offs_inner[None, :]
        else:
            offs_k_scale = MX_SCALE_BLOCK_K * pid_k + tl.arange(0, MX_SCALE_BLOCK_K)
            offs_n_scale = offs_w_n
            # K dimension must be the last dimension for the scales
            MxScalePtrs = MxScale + offs_k_scale.to(index_type)[None, :] * stride_mx_k + offs_n_scale.to(index_type)[:, None] * stride_mx_n
    else:
        MxScalePtrs = None
        offs_k_scale = None
        W_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K: tl.constexpr = 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K

    offs_w_k = PACKED_BLOCK_K_W * pid_k + tl.arange(0, PACKED_BLOCK_K_W)
    W += expt_id * stride_w_e
    WPtrs = W + (offs_w_k.to(index_type)[:, None] * stride_w_k + offs_w_n.to(index_type)[None, :] * stride_w_n)
    # compute output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, BLOCK_K * pid_k, -(BLOCK_K * SPLIT_K)):
        if EVEN_K:
            mask_k = tl.full([BLOCK_K], True, dtype=tl.int1)
            mask_k_w = tl.full([PACKED_BLOCK_K_W], True, dtype=tl.int1)
            mask_k_scale = tl.full([MX_SCALE_BLOCK_K], True, dtype=tl.int1)
        else:
            mask_k = offs_k < k
            mask_k_w = offs_w_k < tl.cdiv(k, W_PACK_DIVISOR)
            if is_microscaled_format and not SWIZZLE_MX:
                mask_k_scale = offs_k_scale < tl.cdiv(k, MX_PACK_DIVISOR)

        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_k_w[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER)
        if is_microscaled_format:
            x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
            mx_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)
            if x_format == "fp16" or x_format == "bf16":
                x_scales: tl.constexpr = None
            else:
                # Scale of 1 in E8M0 format
                x_scales = tl.full((BLOCK_M, BLOCK_K // MX_PACK_DIVISOR), 127, dtype=tl.uint8)
            if SWIZZLE_MX:
                w_scales = _unswizzle_mx_block(tl.load(MxScalePtrs))
            else:
                w_scales = tl.load(MxScalePtrs, mask=mask_k_scale[None, :], other=0.0)
            acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, mx_format, acc=acc, fast_math=True)
            if SWIZZLE_MX:
                MxScalePtrs += (MX_SCALE_BLOCK_K // 4 * SPLIT_K) * stride_mx_k
            else:
                MxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_mx_k
        else:
            acc = tl.dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k
    # bias + scale
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
    else:
        bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
    if Betas is not None:
        betas = tl.load(Betas + start_m + offs_m, mask=mask_m, other=0.0)
    else:
        betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
    else:
        gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    # flexpoint
    x_scale = load_scale(XScale)
    if PER_BATCH_SCALE:
        w_scale = load_scale(WScale + expt_id)
    else:
        w_scale = load_scale(WScale)
    acc *= x_scale * w_scale
    acc = acc + bias[None, :] * betas[:, None]
    acc *= gammas[:, None]
    if out_alpha is not None:
        acc *= out_alpha
    # write-back
    Y += start_z.to(index_type) * stride_y_z
    if WriteBackIndx is not None:
        WriteBackIndx += start_m
        dst_idx = tl.load(WriteBackIndx + offs_m, mask=start_m + offs_m < writeback_size, other=-1)
        mask_m = mask_m & (dst_idx != -1)
        offs_y_m = dst_idx
    else:
        Y += start_m * stride_y_m
        offs_y_m = offs_m

    YPtrs = Y + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    acc = float_to_flex(acc, YExpectedScale, YActualScale, YChecksumScale, mask, Y, FLEXPOINT_SATURATE_INF)
    if EPILOGUE_FN is not None:
        acc = EPILOGUE_FN(acc, *epilogue_fn_args, target_dtype=YPtrs.dtype.element_ty)
    tl.store(YPtrs, acc, mask=mask)


# Imagine N_EXPTS_ACT = 4, n_final_rows = 5, and n_scratchpad_rows = 8.
# Also imagine scatter_indx.src_indx is:
#                   (number of active experts per final row)
#   -1 -1  0 -1     1
#   -1  2 -1 -1     1
#    1  3 -1 -1     2
#   -1  4  5  6     3
#   -1 -1 -1 -1     0 (this row is unused)
#
# Then, row 0 and 1 can be written directly to the final tensor.
# In this case, WriteBackIndx looks like:
#    [0] = 0      : intermediate row 0 is written directly to final row 0
#    [1] = 5+1=6  : scratchpad starts at offset 5
#    [2] = 1      : intermediate row 2 is written directly to final row 1
#    [3] = 5+3=8
#    [4] = 5+4=9
#    [5] = 5+5=10
#    [6] = 5+6=11
#    [7] = -1     : unused (there are only seven intermediate rows)
@triton.jit
def _compute_writeback_idx(
    WriteBackIndx,
    FinalizeScatterIdxs,
    ScatterDstIndx, ScatterSrcIndx,
    n_final_rows, n_scratchpad_rows,
    BLOCK_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
):
    tl.static_assert(N_EXPTS_ACT > 1)

    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_scratchpad_rows
    dst_idxs = tl.load(ScatterDstIndx + offs_m, mask=mask_m, other=-1)
    # Load corresponding rows in ScatterSrcIndx.
    mask = dst_idxs != -1
    src_offs = (dst_idxs // N_EXPTS_ACT) * N_EXPTS_ACT
    src_offs = src_offs[:, None] + tl.arange(0, N_EXPTS_ACT)[None, :]
    src_idxs = tl.load(ScatterSrcIndx + src_offs, mask=mask[:, None], other=-1)
    # Compute the number of actually active experts.
    is_src_active = (src_idxs != -1).to(tl.int32)
    has_one_active = tl.sum(is_src_active, axis=1) == 1
    # Compute the writeback index.
    wb_idx = tl.where(has_one_active, dst_idxs // N_EXPTS_ACT, n_final_rows + offs_m)
    wb_idx = tl.where(mask, wb_idx, -1)
    tl.store(WriteBackIndx + offs_m, wb_idx, mask=mask_m)

    if pid_m >= ((n_final_rows + BLOCK_M - 1) // BLOCK_M):
        return

    mask_m = offs_m < n_final_rows
    src_offs = offs_m[:, None] * N_EXPTS_ACT + tl.arange(0, N_EXPTS_ACT)[None, :]
    src_idxs = tl.load(ScatterSrcIndx + src_offs, mask=mask_m[:, None], other=-1)
    is_src_active = (src_idxs != -1).to(tl.int32)
    has_one_active = tl.sum(is_src_active, axis=1) == 1

    need_finalize_scatter = mask_m and not has_one_active
    finalize_scatter_count = tl.sum(need_finalize_scatter.to(tl.int32))
    if finalize_scatter_count == 0:
        return
    pp_off = tl.atomic_add(FinalizeScatterIdxs + n_final_rows + n_scratchpad_rows, finalize_scatter_count)

    # need_finalize_scatter = [1, 0, 0, 1, 1, 0, 1, 0, 1]
    # arange = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    arange = tl.arange(0, BLOCK_M)
    # idxs = [0, _, _, 3, 4, _, 6, _, 8]
    last = BLOCK_M - 1
    idxs = tl.where(need_finalize_scatter, arange, last)
    # idxs = [0, 3, 4, 6, 8, _, _, _, _]
    idxs = tl.sort(idxs)
    # r = offs_m
    # d = [r[0], r[3], r[4], r[6], r[8], r[-1], r[-1], r[-1], r[-1]]
    d = tl.gather(offs_m, idxs, axis=0)
    s = tl.gather(src_idxs, idxs.expand_dims(1).broadcast_to(src_idxs.shape), axis=0)
    # store destination indices
    Ptr = FinalizeScatterIdxs + pp_off
    tl.store(Ptr + arange, d, mask=arange < finalize_scatter_count)
    # store src indices
    Ptr = FinalizeScatterIdxs + n_final_rows + pp_off * N_EXPTS_ACT
    tl.store(Ptr + N_EXPTS_ACT * arange[:, None] + tl.arange(0, N_EXPTS_ACT)[None, :], s, mask=(arange < finalize_scatter_count)[:, None])
