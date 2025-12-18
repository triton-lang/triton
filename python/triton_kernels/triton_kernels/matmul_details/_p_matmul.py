# isort: off
# fmt: off
import torch
import triton
import triton.language as tl
from triton.tools.ragged_tma import load_ragged, store_ragged
from triton_kernels import target_info
from triton_kernels.tensor_details.layout_details.blackwell_scale import unswizzle_mx_scale_bw, unswizzle_act_mx_scale_bw
from triton_kernels.numerics_details.flexpoint import (
    float_to_flex,
    load_scale,
    nan_propagating_absmax_reduce,
    compute_scale,
)
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE
from triton_kernels.tensor_details.layout_details.hopper_scale import unswizzle_mxfp4_scale_hopper
from triton_kernels.tensor_details.layout_details.hopper_value import mxfp4_to_bf16_triton
from ._common import (
    compute_offsets,
    get_scaled_dot_format_string,
    make_matmul_repr,
    matmul_launch_metadata,
    compute_pids,
)


@triton.constexpr_function
def cuda_capability_geq(major, minor):
    return target_info.cuda_capability_geq(major, minor)

@triton.constexpr_function
def get_dtype(tensor_or_desc: tl.tensor | tl.tensor_descriptor) -> tl.dtype:
    if isinstance(tensor_or_desc, tl.tensor):
        return tensor_or_desc.dtype.element_ty
    elif isinstance(tensor_or_desc, tl.tensor_descriptor):
        return tensor_or_desc.dtype
    else:
        raise ValueError(f"Invalid type: {type(tensor_or_desc)}")

@triton.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = tl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)


_matmul_repr = make_matmul_repr("_p_matmul", [0, 1, 2])
@triton.jit(do_not_specialize=["TOKENS_PER_EXPT_FOR_ANNOTATION"],
            repr=_matmul_repr, launch_metadata=matmul_launch_metadata)
def _p_matmul(
             Y, YPtr, stride_y_k, stride_y_z, stride_y_m, stride_y_n,
             YExpectedScale, YActualScale, YChecksumScale,
             stride_y_mx_k, stride_y_mx_z, stride_y_mx_m, stride_y_mx_n,
             X, XPtr, stride_x_z, stride_x_m, stride_x_k, X_TRANSPOSE: tl.constexpr,
             XScale,
             XMxScale, stride_x_mx_z, stride_x_mx_m, stride_x_mx_k,
             W, WPtr, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: tl.constexpr,
             WScale,
             WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
             OutAcc, stride_acc_z, stride_acc_m, stride_acc_n,
             OutAccScale, Y_ACC_IS_Y: tl.constexpr,
             B, stride_b_e, # Bias
             M, N, K, K_W, # shapes
             # expt data
             Betas, Gammas,
             GatherIndx,
             WriteBackIndx, writeback_size,
             RAGGED_DIMENSION: tl.constexpr,
             XSliceSizes, XSliceOffs, XBlockOffs, XBlockSchedule, X_EXPECTED_SLICE_SIZE: tl.constexpr, X_SLICE_SIZES_DIVISIBILITY: tl.constexpr,
             WSliceSizes, WSliceOffs, WBlockOffs, WBlockSchedule, W_EXPECTED_SLICE_SIZE: tl.constexpr, W_SLICE_SIZES_DIVISIBILITY: tl.constexpr,
             # true grid size
             batch_size, grid_m, grid_n,
             # Out scale
             out_alpha,
             # fused activation function
             ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
             # epilogue transform
             EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
             # MoE config
             N_SLICES: tl.constexpr,
             # precision config
             MAX_NUM_IMPRECISE_ACC: tl.constexpr, ALLOW_TF32: tl.constexpr,
             FLEXPOINT_SATURATE_INF: tl.constexpr,
             PER_BATCH_W_SCALE: tl.constexpr,
             PER_BATCH_OUT_SCALE: tl.constexpr,
             PER_BATCH_ACC_SCALE: tl.constexpr,
             # optimization config
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr,
             # NYI: Must be None
             SWIZZLE_MX_VALUE: tl.constexpr,
             # One of ["BLACKWELL", None]
             SWIZZLE_MX_SCALE: tl.constexpr,
             EPILOGUE_SUBTILE: tl.constexpr,
             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
             W_CACHE_MODIFIER: tl.constexpr,
             NUM_SMS: tl.constexpr,
             X_TMA_MODE: tl.constexpr,
             Y_TMA_MODE: tl.constexpr,
             TOKENS_PER_EXPT_FOR_ANNOTATION=None,
             UPCAST_INDICES: tl.constexpr=False,
             SWAP_XW: tl.constexpr = False,
             IS_EPILOGUE_QUANT_MXFP8: tl.constexpr = False,
             pYPtrs=None,
             map_dst_coord=None,
             all_writes_issued=None,
             reduce_rank=0,
             n_reduce_shards: tl.constexpr = 1,
             ):
    # tl.static_assert(SWIZZLE_MX_VALUE is None, "NYI. Value swizzling")

    # why is this faster than using host-side tensor descriptor?!
    if Y_TMA_MODE is not None:
        Y = tl.make_tensor_descriptor(YPtr, Y.shape, Y.strides[:-1] + (1,), Y.block_shape)

    w_type: tl.constexpr = get_dtype(W)
    is_w_microscaled: tl.constexpr = WMxScale is not None
    is_x_microscaled: tl.constexpr = XMxScale is not None
    is_w_mxfp4: tl.constexpr = w_type == tl.uint8 and is_w_microscaled
    tl.static_assert(not is_w_microscaled or W_TRANSPOSE, "NYI. Non-transposed mxfp4 weights")
    MX_PACK_DIVISOR: tl.constexpr = MXFP_BLOCK_SIZE
    if is_w_microscaled:
        tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                         "mx_weight_ptr must be uint8 or fp8")
        tl.static_assert(get_dtype(WMxScale) == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")

        # We have pack 2 fp4 values in a byte
        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR
        if SWIZZLE_MX_VALUE == "HOPPER_VALUE":
            tl.static_assert(is_w_mxfp4, "Only mxfp4 is supported for HOPPER swizzling")
            tl.static_assert(not is_x_microscaled)
            # We have pack 2 fp4 values in a byte but we divide the dimension by 2
            # when swizzling
            W_K_DIVISOR: tl.constexpr = 1
            W_K_MULTIPLIER: tl.constexpr = 2
            W_N_DIVISOR: tl.constexpr = 4
        else:
            # We have pack 2 fp4 values in a byte
            W_K_DIVISOR: tl.constexpr = 2 if is_w_mxfp4 else 1
            W_K_MULTIPLIER: tl.constexpr = 1
            W_N_DIVISOR: tl.constexpr = 1

        if W_TRANSPOSE:
            # When weight is transposed, 2 fp4 values are packed per Byte along
            # the contiguous dimension, K.
            PACKED_BLOCK_K_W: tl.constexpr = (BLOCK_K // W_K_DIVISOR) * W_K_MULTIPLIER
            PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_N_DIVISOR
        else:
            # When weight is not transposed, fp4 values are *not* packed along
            # the contiguous dimension, N.
            PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
            PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_K_DIVISOR
    else:
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
        PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N
        tl.static_assert(SWIZZLE_MX_SCALE is None)
    if is_x_microscaled:
        x_type: tl.constexpr = get_dtype(X)
        tl.static_assert(x_type == tl.float8e4nv, "mx_act_ptr must be float8e4nv")
        tl.static_assert(get_dtype(XMxScale) == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")
    is_out_microscaled: tl.constexpr = stride_y_mx_z is not None

    if RAGGED_DIMENSION == "M":
        useful_grid_m = tl.load(XBlockOffs + N_SLICES)
    else:
        useful_grid_m = grid_m

    index_type: tl.constexpr = tl.int64

    USE_FLEXPOINT_SCALE: tl.constexpr = YActualScale is not None or YChecksumScale is not None
    HAS_SCATTER: tl.constexpr = WriteBackIndx is not None
    HAS_GATHER: tl.constexpr = GatherIndx is not None
    USE_GATHER_TMA: tl.constexpr = HAS_GATHER and X_TMA_MODE == "dense"
    USE_SCATTER_TMA: tl.constexpr = HAS_SCATTER and Y_TMA_MODE == "dense"

    if RAGGED_DIMENSION == "K":
        tl.static_assert((OutAcc is None) or Y_ACC_IS_Y, "Using differernt y_acc is not supported with TMA kernel.")
        tl.static_assert(not (HAS_SCATTER or USE_GATHER_TMA or USE_SCATTER_TMA), "Cannot be used with RAGGED_DIMENSION == 'K'")

    if EPILOGUE_SUBTILE is None:
        SUBTILE_FACTOR: tl.constexpr = 1
    else:
        SUBTILE_FACTOR: tl.constexpr = EPILOGUE_SUBTILE
    EPILOGUE_BLOCK_N: tl.constexpr = BLOCK_N // SUBTILE_FACTOR
    OUT_BLOCK_N: tl.constexpr = EPILOGUE_BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    num_blocks = batch_size * useful_grid_m * grid_n * SPLIT_K

    # If true, do not share loop-carried variables between the prologue and the
    # epilogue to enable better pipelining with mmav5
    INDEPENDENT_EPILOGUE: tl.constexpr = cuda_capability_geq(10, 0)

    # start negative; will be incremented at the top of the loop
    if INDEPENDENT_EPILOGUE:
        tile_id1 = tl.program_id(0) - NUM_SMS

    # Keep track of local max for updating flexpoint scales.
    USE_LOCAL_ABSMAX: tl.constexpr = (YActualScale is not None) and (not PER_BATCH_OUT_SCALE) and (not is_out_microscaled) and (pYPtrs is None)
    if USE_LOCAL_ABSMAX:
        THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
        local_absmax = tl.full([THREADS_PER_BLOCK], 0.0, tl.uint32)

    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr = is_w_microscaled and BLOCK_M * BLOCK_N >= 128 * 256

    for block_id in tl.range(tl.program_id(0), num_blocks, NUM_SMS, flatten=True, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER, warp_specialize=True):

        pid_z, pid_m, pid_n, pid_k = compute_pids(block_id, useful_grid_m, grid_n, num_blocks, XCD_SWIZZLE, GROUP_M, SPLIT_K)

        # ------------------------------------------------------------
        # prologue
        # ------------------------------------------------------------
        off_w_z, off_x_z, off_y_z, slice_off_m, slice_block_off_m, off_m, off_k_x0, off_k_w0 = compute_offsets(
            pid_z, pid_m, pid_k,
            XBlockSchedule, XSliceOffs, XBlockOffs, X_SLICE_SIZES_DIVISIBILITY,
            WBlockSchedule, WSliceOffs, W_SLICE_SIZES_DIVISIBILITY,
            RAGGED_DIMENSION,
            BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W, SPLIT_K
        )

        # TODO: if RAGGED_DIMENSION == "M"
        if RAGGED_DIMENSION == "M":
            shape_m = tl.load(XSliceSizes + off_w_z)
        else:
            shape_m = M
        off_n = BLOCK_N * pid_n
        off_w_n = PACKED_BLOCK_N_W * pid_n

        # ---- offset x ------
        if USE_GATHER_TMA:
            offs_m = off_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < shape_m
            if XBlockSchedule is None:
                offs_x_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m, mask=mask_m)
                # Bump rows to account for the Z offset.
                offs_x_m += off_x_z * (stride_x_z // stride_x_m)
                offs_x_m = tl.where(mask_m, offs_x_m, -1)
            else:
                offs_x_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m, mask=mask_m, other=-1)
        if X_TMA_MODE is None:
            XBase = X + off_x_z.to(index_type) * stride_x_z
            offs_m = off_m + tl.arange(0, BLOCK_M)
            offs_m = tl.max_contiguous(tl.multiple_of(offs_m % shape_m, BLOCK_M), BLOCK_M)
            # no needs to bounds-check here because `offs_m` wraps around M dim
            if GatherIndx is not None:
                tl.static_assert(HAS_GATHER)
                offs_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m)
            offs_x_m = offs_m.to(index_type)[:, None] * stride_x_m
            offs_x_k = (off_k_x0.to(index_type) + tl.arange(0, BLOCK_K))[None, :] * stride_x_k

        XMxScalePtrs = None
        if is_x_microscaled and stride_x_mx_z is not None: # x is mx but not using TMA
            offs_m = off_m + tl.arange(0, BLOCK_M)
            XMxScalePtrs = XMxScale + off_x_z.to(index_type) * stride_x_mx_z
            if GatherIndx is None:
                XMxScalePtrs += slice_off_m * stride_x_mx_m
            offs_k_scale = off_k_x0 // MXFP_BLOCK_SIZE + tl.arange(0, MX_SCALE_BLOCK_K)
            XMxScalePtrs += (offs_x_m if USE_GATHER_TMA else offs_m).to(index_type)[:, None] * stride_x_mx_m
            XMxScalePtrs += offs_k_scale.to(index_type)[None, :] * stride_x_mx_k

        acc = tl.zeros((BLOCK_N, BLOCK_M) if SWAP_XW else (BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ------------------------------------------------------------
        # inner loop
        # ------------------------------------------------------------
        loop_k = tl.load(XSliceSizes + pid_z) if RAGGED_DIMENSION == "K" else K - off_k_x0
        k_tiles = tl.cdiv(loop_k, BLOCK_K * SPLIT_K)
        loop_bound = tl.maximum(k_tiles, 1)
        tl.assume(loop_bound > 0)  # Currently necessary for the compiler to flatten the loop properly.
        for ki in tl.range(loop_bound, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
            if RAGGED_DIMENSION == "K" and ki >= k_tiles:
                # Tile #ki does not exist: use out-of-bound indices to mask all loads.
                off_k_x = K
                off_k_w = K_W
            else:
                off_k_x = off_k_x0 + ki * BLOCK_K * SPLIT_K
                off_k_w = off_k_w0 + ki * PACKED_BLOCK_K_W * SPLIT_K

            # --- load x ---
            if USE_GATHER_TMA:
                x = X.gather(offs_x_m, off_k_x)
            elif X_TMA_MODE == "dense":
                if X_TRANSPOSE:
                    x = X.load([off_x_z, off_k_x, slice_off_m + off_m])
                    x = x.reshape(BLOCK_K, BLOCK_M).T
                else:
                    x = X.load([off_x_z, slice_off_m + off_m, off_k_x])
                    x = x.reshape(BLOCK_M, BLOCK_K)
            elif X_TMA_MODE == "ragged":
                x = load_ragged(X, slice_off_m, shape_m, [off_x_z, off_m, off_k_x], ragged_dim=1)
                x = x.reshape(BLOCK_M, BLOCK_K)
            else:
                tl.static_assert(X_TMA_MODE is None)
                XPtrs = XBase + offs_x_m + offs_x_k
                XBase += BLOCK_K * SPLIT_K * stride_x_k
                mask_k = tl.arange(0, BLOCK_K) < K - off_k_x
                if EVEN_K:
                    if SPLIT_K > 1:
                        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
                    else:
                        x = tl.load(XPtrs)
                else:
                    x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)

            # --- load x_scale ---
            x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
            if is_x_microscaled:
                if XMxScalePtrs is not None: # not using TMA for x scale load
                    # dividing MX_PACK_DIVISOR by W_K_DIVISOR because off_k_w is
                    # already divided by W_K_DIVISOR (2 for mxfp4 where 2 fp4
                    # values are packed per Byte along K)
                    off_k_mx = off_k_w // (MX_PACK_DIVISOR // W_K_DIVISOR)
                    if EVEN_K:
                        mask_k_scale = tl.full([MX_SCALE_BLOCK_K], True, dtype=tl.int1)
                    else:
                        mask_k_scale = off_k_mx + tl.arange(0, MX_SCALE_BLOCK_K) < tl.cdiv(K, MX_PACK_DIVISOR)
                    mask_m = off_m + tl.arange(0, BLOCK_M) < shape_m
                    x_scales = tl.load(XMxScalePtrs, mask=mask_k_scale[None, :] & mask_m[:, None], other=0.0)
                else: # use TMA for x scale load - only cover batched case for now
                    if X_TMA_MODE == "dense":
                        off_m_scale = off_x_z * ((M + 127) // 128) + off_m // 128
                    else:
                        # slice_block_off_m points to the start of the current slice in the padded version
                        # + off_m points to the current block in the slice
                        off_m_scale = slice_block_off_m + off_m // 128
                    x_scales = XMxScale.load([0, off_m_scale, off_k_x // MX_PACK_DIVISOR // 4, 0, 0])
                    x_scales = unswizzle_act_mx_scale_bw(x_scales)
            elif x_format == "fp16" or x_format == "bf16":
                x_scales: tl.constexpr = None
            else:
                x_scales = tl.full((BLOCK_M, BLOCK_K // MX_PACK_DIVISOR), 127, dtype=tl.uint8)

            # --- load w ---
            if W_TRANSPOSE:
                w = tl.reshape(W.load([off_w_z, off_w_n, off_k_w]), W.block_shape[1:]).T
            else:
                w = tl.reshape(W.load([off_w_z, off_k_w, off_w_n]), W.block_shape[1:])

            # --- load w_scale ---
            w_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)
            if is_w_microscaled:
                off_k_mx = off_k_w // (MX_PACK_DIVISOR // W_K_DIVISOR)
                tl.static_assert(MX_PACK_DIVISOR % W_K_DIVISOR == 0)
                if SWIZZLE_MX_SCALE == "BLACKWELL_SCALE":
                    flattened_expt_n_idx = off_w_z * ((N + 127) // 128) + (off_n // 128)
                    w_scales = WMxScale.load([0, flattened_expt_n_idx, off_k_mx // 4, 0, 0])
                    w_scales = w_scales.reshape((w_scales.shape[1], w_scales.shape[2] * w_scales.shape[-2] * w_scales.shape[-1]))
                    w_scales = unswizzle_mx_scale_bw(w_scales)
                elif SWIZZLE_MX_SCALE == "HOPPER_SCALE":
                    # NYI: Hopper swizzling with non-transposed W
                    tl.static_assert(W_TRANSPOSE)
                    off_n_scale = pid_n * (BLOCK_N // 32)
                    off_k_scale = (off_k_w // PACKED_BLOCK_K_W) * MX_SCALE_BLOCK_K * 32
                    w_scales = WMxScale.load([off_w_z, off_n_scale, off_k_scale])
                    w_scales = tl.reshape(w_scales, *w_scales.shape[1:])
                    num_warps: tl.constexpr = tl.extra.cuda.num_warps()
                    w_scales = unswizzle_mxfp4_scale_hopper(w_scales, mx_axis=1, num_warps=num_warps)
                else:
                    w_scales = WMxScale.load([off_w_z, off_k_mx, off_n])
                    w_scales = tl.reshape(w_scales, *w_scales.shape[1:]).T

            # --- update accumulator ---
            if is_w_microscaled:
                if SWIZZLE_MX_VALUE == "HOPPER_VALUE":
                    tl.static_assert(x_format == "bf16")
                    tl.static_assert(w_format == "e2m1")
                    tl.static_assert(SWAP_XW)
                    wT = mxfp4_to_bf16_triton(w.T, w_scales, mx_axis=1)
                    tl.static_assert(wT.dtype == tl.bfloat16)
                    acc = tl.dot(wT, x.T, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                else:
                    if SWAP_XW:
                        acc = tl.dot_scaled(w.T, w_scales, w_format, x.T, x_scales, x_format, acc=acc, fast_math=True)
                    else:
                        acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, w_format, acc=acc, fast_math=True)
            else:
                if SWAP_XW:
                    acc = tl.dot(w.T, x.T, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                else:
                    acc = tl.dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)

            if is_x_microscaled and XMxScalePtrs is not None:
                XMxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_x_mx_k

        # ------------------------------------------------------------
        # epilogue
        # ------------------------------------------------------------
        if INDEPENDENT_EPILOGUE:
            tile_id1 += NUM_SMS
            pid_s1, pid_m1, pid_n1, pid_k1 = compute_pids(tile_id1, useful_grid_m, grid_n, num_blocks, XCD_SWIZZLE, GROUP_M, SPLIT_K)
            expt_id1, _, start_z1, start_m1, _, off_m1, _, _ = compute_offsets(
                pid_z, pid_m, pid_k,
                XBlockSchedule, XSliceOffs, XBlockOffs, X_SLICE_SIZES_DIVISIBILITY,
                WBlockSchedule, WSliceOffs, W_SLICE_SIZES_DIVISIBILITY,
                RAGGED_DIMENSION,
                BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W, SPLIT_K
            )
            off_n1 = pid_n1 * BLOCK_N
            if RAGGED_DIMENSION == "M":
                eM1 = tl.load(XSliceSizes + expt_id1)
            else:
                eM1 = M
        else:
            tile_id1, expt_id1, start_z1, start_m1, eM1 = block_id, off_w_z, off_y_z, slice_off_m, shape_m
            off_m1, off_n1, pid_k1 = off_m, off_n, pid_k

        offs_m = off_m1 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < eM1
        if USE_SCATTER_TMA:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE
            if SPLIT_K > 1:
                # Compute the split k offset in number of rows, and add it to offs_y_m.
                # This allows us to write to the correct slice in the output tensor while using
                # a 2D TMA scatter.
                tl.device_assert(stride_y_k // stride_y_m == tl.cdiv(stride_y_k, stride_y_m))
                split_k_row_offs = pid_k1 * (stride_y_k // stride_y_m)
                offs_y_m = tl.where(mask_m, offs_y_m + split_k_row_offs, offs_y_m)
        elif Y_TMA_MODE is None and HAS_SCATTER:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE
        else:
            offs_y_m = start_m1 + offs_m
            MASK_ACC = False if USE_GATHER_TMA else USE_FLEXPOINT_SCALE

        # bias + scale
        offs_y_n = off_n1 + tl.arange(0, BLOCK_N)
        mask_n = offs_y_n < N
        if B is not None:
            BPtrs = B + expt_id1 * stride_b_e + offs_y_n
            if pid_k1 == 0:
                bias = tl.load(BPtrs, mask=mask_n, other=0)
            else:
                bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        if Betas is not None:
            betas = tl.load(Betas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        if Gammas is not None:
            gammas = tl.load(Gammas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        x_scale = load_scale(XScale)
        if PER_BATCH_W_SCALE:
            w_scale = load_scale(WScale + expt_id1)
        else:
            w_scale = load_scale(WScale)

        accs = (acc,)
        biases = (bias,)

        if SUBTILE_FACTOR >= 2:
            if SWAP_XW:
                acc = acc.reshape(2, BLOCK_N // 2, BLOCK_M).permute(1, 2, 0)
            else:
                acc = acc.reshape(BLOCK_M, 2, BLOCK_N // 2).permute(0, 2, 1)
            acc0, acc1 = acc.split()
            accs = (acc0, acc1)
            bias0, bias1 = bias.reshape(2, BLOCK_N // 2).permute(1, 0).split()
            biases = (bias0, bias1)

        if SUBTILE_FACTOR >= 4:
            if SWAP_XW:
                acc0 = acc0.reshape(2, BLOCK_N // 4, BLOCK_M).permute(1, 2, 0)
                acc1 = acc1.reshape(2, BLOCK_N // 4, BLOCK_M).permute(1, 2, 0)
            else:
                acc0 = acc0.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1)
                acc1 = acc1.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1)
            acc00, acc01 = acc0.split()
            acc10, acc11 = acc1.split()
            accs = (acc00, acc01, acc10, acc11)
            bias00, bias01 = bias0.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            bias10, bias11 = bias1.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            biases = (bias00, bias01, bias10, bias11)

        tl.static_assert(EPILOGUE_BLOCK_N == BLOCK_N // SUBTILE_FACTOR)
        tl.static_assert(len(accs) == SUBTILE_FACTOR)

        if is_out_microscaled:
            MX_SCALE_BLOCK_N: tl.constexpr = OUT_BLOCK_N // MXFP_BLOCK_SIZE

        for a_i in tl.static_range(len(accs)):
            acc_tile = accs[a_i]
            acc_tile *= x_scale * w_scale

            if SWAP_XW:
                acc_tile = acc_tile.T

            acc_tile = acc_tile + biases[a_i][None, :] * betas[:, None]
            if out_alpha is not None:
                acc_tile *= out_alpha

            if ACTIVATION_FN is not None:
                out = ACTIVATION_FN(acc_tile, *activation_fn_args)
                tl.static_assert(out.shape[1] == OUT_BLOCK_N, f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})")
            else:
                tl.static_assert(ACTIVATION_REDUCTION_N == 1, "Activation reduction must be 1 if no activation fn is provided")
                out = acc_tile

            out *= gammas[:, None]

            if OutAcc is not None:
                tl.static_assert(not USE_SCATTER_TMA)
                out_off_n = off_n1 // ACTIVATION_REDUCTION_N + a_i * OUT_BLOCK_N
                if PER_BATCH_ACC_SCALE:
                    ScalePtr = OutAccScale + start_z1
                else:
                    ScalePtr = OutAccScale

                tl.static_assert(Y_TMA_MODE == "dense" or Y_TMA_MODE is None)
                if Y_TMA_MODE == "dense":
                    off_kz = pid_k * batch_size + start_z1
                    acc = Y.load([off_kz, off_m1, out_off_n])
                    acc = acc.reshape(out.shape)
                    out += acc * load_scale(ScalePtr)
                else:
                    offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                    mask_n = offs_y_n < yN

                    AccPtrs = YPtr + pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_n
                    mask = mask_m[:, None] & mask_n[None, :]
                    acc = tl.load(AccPtrs, mask=mask, other=0.0)
                    out += acc * load_scale(ScalePtr)

            if MASK_ACC:
                out = tl.where(mask_m[:, None], out, 0.0)

            out_off_n = off_n1 // ACTIVATION_REDUCTION_N + a_i * OUT_BLOCK_N
            if is_out_microscaled:
                tl.static_assert(EPILOGUE_FN is not None)
                offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN
                out, out_scale = EPILOGUE_FN(out, mask_m[:, None] & mask_n[None, :], *epilogue_fn_args)
                tl.static_assert(BLOCK_N % MX_SCALE_BLOCK_N == 0, "")
                offs_y_n_scale = off_n1 // ACTIVATION_REDUCTION_N // MXFP_BLOCK_SIZE + a_i * MX_SCALE_BLOCK_N + tl.arange(0, MX_SCALE_BLOCK_N)
                mask_n_scale = offs_y_n_scale < tl.cdiv(yN, MXFP_BLOCK_SIZE)
                offs_y_mx_k = 0
                if USE_SCATTER_TMA:
                    # Convert -1 offsets to INT_MAX. We do this by clearing the leading bit. Note that
                    # there shouldn't be any other negative values.
                    offs_y_mx_z = 0
                    offs_y_mx_m = (offs_y_m.to(tl.uint32, bitcast=True) & 0x7FFFFFFF).to(tl.int32, bitcast=True)
                elif Y_TMA_MODE == "dense":
                    offs_y_mx_z = pid_k * batch_size + start_z1
                    offs_y_mx_m = off_m1 + tl.arange(0, BLOCK_M)
                elif Y_TMA_MODE == "ragged":
                    offs_y_mx_z = pid_k
                    offs_y_mx_m = start_m1 + off_m1 + tl.arange(0, BLOCK_M)
                else:
                    tl.static_assert(Y_TMA_MODE is None)
                    offs_y_mx_k = pid_k1
                    offs_y_mx_z = start_z1
                YActualScalePtrs = YActualScale + offs_y_mx_k.to(index_type) * stride_y_mx_k + offs_y_mx_z.to(index_type) * stride_y_mx_z + offs_y_mx_m.to(index_type)[:, None] * stride_y_mx_m + offs_y_n_scale.to(index_type)[None, :] * stride_y_mx_n
                tl.store(YActualScalePtrs, out_scale, mask=mask_m[:, None] & mask_n_scale[None, :])
            else:
                # Flexpoint
                if USE_LOCAL_ABSMAX:
                    out_view = tl.reshape(out, [out.numel // THREADS_PER_BLOCK, THREADS_PER_BLOCK], can_reorder=True)
                    local_absmax = tl.maximum(local_absmax, nan_propagating_absmax_reduce(out_view, axis=0))

                if PER_BATCH_OUT_SCALE:
                    ExpectedScale = YExpectedScale + start_z1
                    ActualScale = YActualScale + start_z1
                else:
                    ExpectedScale = YExpectedScale
                    ActualScale = None  # local absmax is tracked and updated after the loop

                out = float_to_flex(
                    out, ExpectedScale, ActualScale, YChecksumScale,
                    None, # mask: out is manually masked to 0
                    YPtr, FLEXPOINT_SATURATE_INF
                )
                if EPILOGUE_FN is not None and not IS_EPILOGUE_QUANT_MXFP8:
                    out = EPILOGUE_FN(out, *epilogue_fn_args, target_dtype=YPtr.dtype.element_ty, pid=len(accs)*tile_id1 + a_i)

            out = out.to(YPtr.dtype.element_ty)

            if pYPtrs is None:
                if USE_SCATTER_TMA:
                    # Convert -1 offsets to INT_MAX. We do this by clearing the leading bit. Note that
                    # there shouldn't be any other negative values.
                    offs_y_m = (offs_y_m.to(tl.uint32, bitcast=True) & 0x7FFFFFFF).to(tl.int32, bitcast=True)
                    Y.scatter(out, offs_y_m, out_off_n)
                elif Y_TMA_MODE == "dense":
                    out = tl.reshape(out, [1] + out.shape)
                    off_kz = pid_k * batch_size + start_z1
                    Y.store([off_kz, off_m1, out_off_n], out)
                elif Y_TMA_MODE == "ragged":
                    out = tl.reshape(out, [1] + out.shape)
                    store_ragged(Y, start_m1, eM1, [pid_k, off_m1, out_off_n], out, ragged_dim=1)
                else:
                    tl.static_assert(Y_TMA_MODE is None)
                    offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                    mask_n = offs_y_n < yN
                    mask = mask_m[:, None] & mask_n[None, :]
                    offs_kzmn = pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_n
                    tl.store(YPtr + offs_kzmn, out, mask=mask)
            else:
                tl.static_assert(Y_TMA_MODE is None, "TMA is not supported with fused comms")
                offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN
                mask = mask_m[:, None] & mask_n[None, :]

                dst_shard_idx, dst_y_m, dst_y_n = map_dst_coord.fn(
                    start_m1 + off_m1 if WriteBackIndx is None else None, offs_y_m,
                    out_off_n, offs_y_n,
                    *map_dst_coord.captured)
                offs_kzmn = (
                    pid_k1.to(index_type) * stride_y_k +
                    start_z1.to(index_type) * stride_y_z +
                    dst_y_n[None, :] * stride_y_n +
                    dst_y_m.to(index_type)[:, None] * stride_y_m * n_reduce_shards + reduce_rank * stride_y_m
                )
                for i in tl.static_range(n_reduce_shards):
                    if dst_shard_idx is not None:
                        peer = dst_shard_idx * n_reduce_shards + (reduce_rank + i) % n_reduce_shards
                    else:
                        peer = (reduce_rank + i) % n_reduce_shards
                    peer_Y_ptr = tl.load(pYPtrs + peer).to(tl.pointer_type(YPtr.type.element_ty))
                    if len(peer_Y_ptr.shape) == 0:
                        tl.multiple_of(peer_Y_ptr, 16)
                    else:
                        tl.multiple_of(peer_Y_ptr, [16, 16])
                    tl.store(peer_Y_ptr + offs_kzmn, out, mask=mask)


    # Update the flexpoint scales
    if USE_LOCAL_ABSMAX:
        tl.atomic_max(YActualScale, compute_scale(local_absmax.to(tl.float32, bitcast=True), YPtr), sem="relaxed")

    if pYPtrs is not None:
        all_writes_issued.fn(*all_writes_issued.captured)

_per_device_alloc_fns = {}
def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}
        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if stream not in _per_stream_tensors or _per_stream_tensors[stream].numel() < size:
                _per_stream_tensors[stream] = torch.empty(size, device=device, dtype=torch.int8)
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]
