import functools
import torch
import triton
import triton.language as tl
from triton_kernels import target_info
from triton_kernels.numerics_details.mxfp import _unswizzle_mx_block, get_scaled_dot_format_string
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale, nan_propagating_absmax_reduce, compute_scale
from ._common import make_matmul_repr, matmul_launch_metadata, swizzle2d, xcd_swizzle

# fmt: off

@tl.constexpr_function
def cuda_capability_geq(major, minor):
    return target_info.cuda_capability_geq(major, minor)

# TODO: this is a limitation of the triton frontend
# we shouldn't have to do that!
def inline_function(f):
    """
    Wraps an arbitrary Python function so that it can be inlined into a Triton function at compile-time.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    # disguise the function as a Triton builtin to avoid raising an error
    # that we're calling a non-JIT function from within a Triton kernel:
    wrapper.__triton_builtin__ = True
    wrapper.__module__ = getattr(tl, "__name__", "triton.language")
    return wrapper

@inline_function
def _load_tensor_desc(desc, offs, transpose: tl.constexpr = False, _builder=None):
    if transpose:
        offs = offs[:-2] + [offs[-1], offs[-2]]
    res = desc.load(offs, _builder=_builder)
    res = tl.reshape(res, desc.block_shape[-2:], _builder=_builder)
    if transpose:
        res = tl.trans(res, _builder=_builder)
    return res


# Helper function to recreate a TMA desc with the same fields, but with a new pointer and optional new shape.
@inline_function
def _update_tensor_desc(desc, ptr, shape=None, _builder=None):
    return tl.make_tensor_descriptor(
        ptr,
        shape=shape or desc.shape,
        # last dim must be constexpr 1; reflecting the old descriptor drops the constexpr
        strides=desc.strides[:-1] + [tl.constexpr(1)],
        block_shape=desc.block_shape,
        _builder=_builder,
    )

@triton.jit
def _make_tensor_desc(ptr, shape, strides, block_shape, transpose: tl.constexpr = False):
    tl.static_assert(len(shape) == len(strides))
    tl.static_assert(len(strides) == len(block_shape))
    if transpose:
        return tl.make_tensor_descriptor(
            ptr,
            shape=shape[:-2] + [shape[-1], shape[-2]],
            strides=strides[:-2] + [strides[-1], tl.constexpr(1)],
            block_shape=block_shape[:-2] + [block_shape[-1], block_shape[-2]],
        )
    else:
        return tl.make_tensor_descriptor(
            ptr,
            shape=shape,
            strides=strides[:-1] + [tl.constexpr(1)],
            block_shape=block_shape,
        )

@triton.jit
def _load_tile_attrs(
    tile_id, num_tiles, grid_m, grid_n, padding_m,
    M, ExptData, ExptHist, ExptOffs,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr):
    # unpack and swizzle program ids
    pid_emnk = tile_id
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, num_tiles // SPLIT_K, XCD_SWIZZLE)
    pid_e = pid_emnk // ((grid_m - padding_m) * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % ((grid_m - padding_m) * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, (grid_m - padding_m), grid_n, GROUP_M)

    # unpack expert data
    if ExptData is None:
        tl.static_assert(M is not None)
        expt_id, start_z, start_m, block_id, eM = pid_e, pid_e, 0, pid_m, -1
    else:
        tl.static_assert(M is None)
        expt_data = tl.load(ExptData + pid_m)
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        eM = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        start_z = 0

    off_m = BLOCK_M * block_id
    off_n = BLOCK_N * pid_n

    return expt_id, start_z, start_m, eM, off_m, off_n, pid_k


@triton.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = tl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)


_matmul_ogs_repr = make_matmul_repr("_p_matmul_ogs", [0, 1, 2])
@triton.jit(repr=_matmul_ogs_repr, launch_metadata=matmul_launch_metadata)
def _p_matmul_ogs(
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
             UPCAST_INDICES:tl.constexpr=False,
             DISABLE_Y_TMA: tl.constexpr=False,
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

        # We have pack 2 fp4 values in a byte
        W_PACK_DIVISOR: tl.constexpr = 2 if W.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_PACK_DIVISOR
        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR
    else:
        W_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K: tl.constexpr = 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K

    if ExptOffsSum is not None:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    HAS_FUSED_SCATTER: tl.constexpr = WriteBackIndx is not None
    index_type: tl.constexpr = tl.int64

    # set masked out rows to 0
    if HAS_FUSED_SCATTER and N_EXPTS_ACT == 1:
        # Iterate with reversed pids so that later pids will get more tiles if the number of
        # tiles isn't evenly divisible by the number of SMs.
        # The main loop after this iterates in the forward direction such that earlier
        # pids get more tiles if the number of tiles isn't evenly divisible.
        # This helps balance the work across the SMs.
        for pid_mnk in range(NUM_SMS - tl.program_id(0) - 1, batch_size * grid_m * grid_n * SPLIT_K, NUM_SMS):
            pid_k = pid_mnk % SPLIT_K
            pid_mn = pid_mnk // SPLIT_K
            pid_m, pid_n = swizzle2d(pid_mn, grid_m, grid_n, GROUP_M)

            offs_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
            offs_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
            src_idx = tl.load(ScatterSrcIndx + offs_m, mask=offs_m < num_idxs, other=0)
            YPtrs = Y + offs_m.to(index_type)[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
            mask_n = offs_n < N
            mask = (src_idx == -1)[:, None] & mask_n[None, :]
            tl.store(YPtrs + pid_k * stride_y_k, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), mask=mask)

    USE_FLEXPOINT_SCALE: tl.constexpr = YActualScale is not None or YChecksumScale is not None

    INT_MAX: tl.constexpr = 2147483647
    HAS_TMA_GS: tl.constexpr = cuda_capability_geq(10, 0)
    USE_GATHER_TMA: tl.constexpr = (HAS_TMA_GS and GatherIndx is not None)
    X_USE_LOAD_TMA: tl.constexpr = GatherIndx is None and not USE_GATHER_TMA
    USE_SCATTER_TMA: tl.constexpr = (HAS_TMA_GS and HAS_FUSED_SCATTER) and not DISABLE_Y_TMA

    if USE_GATHER_TMA:
        x_desc = tl.make_tensor_descriptor(
            X,
            # No masking on the M dimension because we manually mask by setting indices to -1
            shape=[INT_MAX, K],
            strides=[stride_x_m, stride_x_k],
            block_shape=[1, BLOCK_K]
        )
    elif X_USE_LOAD_TMA:
        x_desc = tl.make_tensor_descriptor(
            X,
            # When M is ragged, we don't mask the input rows, but mask the accumulator result in the epilogue.
            # So shape[0] here is the global number of rows in the X matrix, which allows using an invariant descriptor.
            shape=[NRows, K],
            strides=[stride_x_m, stride_x_k],
            block_shape=[BLOCK_M, BLOCK_K]
        )

    w_desc = _make_tensor_desc(W,
        shape=[N_EXPTS_TOT if ExptData is not None else batch_size,
            (K + W_PACK_DIVISOR - 1) // W_PACK_DIVISOR, N],
        strides=[stride_w_e, stride_w_k, stride_w_n],
        block_shape=[1, PACKED_BLOCK_K_W, BLOCK_N],
        transpose=W_TRANSPOSE)

    if is_microscaled_format:
        PackedK = (K + MX_PACK_DIVISOR - 1) // MX_PACK_DIVISOR
        if SWIZZLE_MX:
            mx_desc = tl.make_tensor_descriptor(
                MxScale,
                shape=[
                    N_EXPTS_TOT if ExptData is not None else batch_size,
                    (N + 127) // 128, (PackedK + 3) // 4, 32, 4 * 4,
                ],
                strides=[stride_mx_e, stride_mx_n, stride_mx_k, 4 * 4, 1],
                block_shape=[1, BLOCK_N // 128, MX_SCALE_BLOCK_K // 4, 32, 4 * 4]
            )
        else:
            mx_desc = _make_tensor_desc(
                MxScale,
                shape=[N_EXPTS_TOT if ExptData is not None else batch_size, PackedK, N],
                strides=[stride_mx_e, stride_mx_k, stride_mx_n],
                block_shape=[1, MX_SCALE_BLOCK_K, BLOCK_N],
                transpose=MX_TRANSPOSE
            )

    EPILOGUE_BLOCK_N: tl.constexpr = BLOCK_N // 2 if EPILOGUE_SUBTILE else BLOCK_N

    if USE_SCATTER_TMA:
        y_desc = tl.make_tensor_descriptor(
            Y,
            # No masking on the M dimension because we manually mask by setting indices to INT_MAX
            shape=[INT_MAX - 1, N],
            strides=[stride_y_m, stride_y_n],
            block_shape=[1, EPILOGUE_BLOCK_N],
        )

    k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    num_tiles = batch_size * (grid_m - padding_m) * grid_n * SPLIT_K

    # If true, do not share loop-carried variables between the prologue and the
    # epilogue to enable better pipelining with mmav5
    INDEPENDENT_EPILOGUE: tl.constexpr = cuda_capability_geq(10, 0)

    # start negative; will be incremented at the top of the loop
    if INDEPENDENT_EPILOGUE:
        tile_id1 = tl.program_id(0) - NUM_SMS

    # Keep track of local max for updating flexpoint scales.
    THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
    local_absmax = tl.full([THREADS_PER_BLOCK], 0.0, tl.uint32)

    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr = is_microscaled_format and BLOCK_M * BLOCK_N >= 128 * 256
    # Enable warp specialization when all loads are TMA loads. Don't enable it
    # for mixed-precision yet.
    ENABLE_WS: tl.constexpr = True
    WARP_SPECIALIZE: tl.constexpr = (USE_GATHER_TMA or X_USE_LOAD_TMA) and ENABLE_WS

    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER, warp_specialize=WARP_SPECIALIZE):
        expt_id, start_z, start_m, eM, off_m, off_n, pid_k = _load_tile_attrs(
            tile_id, num_tiles, grid_m, grid_n, padding_m,
            M, ExptData, ExptHist, ExptOffs,
            BLOCK_M, BLOCK_N, SPLIT_K,
            GROUP_M, XCD_SWIZZLE)

        # Base pointers and offsets. These will be DCE'ed if unused in the TMA path.
        XBase = X + start_z.to(index_type) * stride_x_z
        offs_x_k = tl.arange(0, BLOCK_K)[None, :] * stride_x_k
        if SPLIT_K > 1:
            offs_x_k += pid_k.to(index_type) * BLOCK_K * stride_x_k
        offs_w_n = off_n + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)

        # If the operands are swapped, the TMA layout of the MX scales are not optimal for the weights anymore.
        # The scales will be loaded with normal loads instead.
        if is_microscaled_format and SWAP_XW:
            offs_mx_k = tl.arange(0, MX_SCALE_BLOCK_K)

            PACKED_MX_BLOCK: tl.constexpr = (MX_SCALE_BLOCK_K // 4) * 32 * 4 * 4
            offs_mx_inner = tl.arange(0, PACKED_MX_BLOCK)
            offs_mx_outer = ((off_n // 128) + tl.arange(0, BLOCK_N // 128)) % N
            offs_mx_outer = tl.max_contiguous(tl.multiple_of(offs_mx_outer, BLOCK_N // 128), BLOCK_N // 128)

            if SPLIT_K > 1:
                offs_mx_k += MX_SCALE_BLOCK_K * pid_k
                offs_mx_inner += PACKED_MX_BLOCK * pid_k

        if X_USE_LOAD_TMA:
            if ExptData is None:
                # start_z may change; update the descriptor
                x_desc = _update_tensor_desc(x_desc, XBase)
        else:
            offs_m = off_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < (M if M is not None else eM)
            if USE_GATHER_TMA:
                # Mask the gather indices and load -1 instead. TMA will handle OOB accesses.
                offs_x_m = tl.load(GatherIndx + start_m.to(index_type) + offs_m,
                                   mask=mask_m, other=-N_EXPTS_ACT) // N_EXPTS_ACT
                if ExptData is None:  # start_z may change; update the descriptor
                    x_desc = _update_tensor_desc(x_desc, XBase)
            else:
                if M is not None:
                    offs_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
                else:
                    offs_m = tl.max_contiguous(tl.multiple_of(offs_m % eM, BLOCK_M), BLOCK_M)
                # no needs to bounds-check here because `offs_m` wraps around M dim
                offs_m = tl.load(GatherIndx + start_m.to(index_type) + offs_m) // N_EXPTS_ACT
                offs_x_m = offs_m.to(index_type)[:, None] * stride_x_m

        acc = tl.zeros((BLOCK_N, BLOCK_M) if SWAP_XW else (BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in tl.range(k_tiles, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
            off_k = pid_k * BLOCK_K + ki * BLOCK_K * SPLIT_K
            off_k_w = pid_k * PACKED_BLOCK_K_W + ki * PACKED_BLOCK_K_W * SPLIT_K
            off_k_mx = pid_k * MX_SCALE_BLOCK_K + ki * MX_SCALE_BLOCK_K * SPLIT_K

            if USE_GATHER_TMA:
                x = x_desc.gather(offs_x_m, off_k)
            elif X_USE_LOAD_TMA:
                x = x_desc.load([start_m + off_m, off_k])
            else:
                XPtrs = XBase + offs_x_m + offs_x_k
                XBase += BLOCK_K * SPLIT_K * stride_x_k
                if EVEN_K:
                    if SPLIT_K > 1:
                        x = tl.load(XPtrs, mask=off_k < K, other=0.0)
                    else:
                        x = tl.load(XPtrs)
                else:
                    mask_k = tl.arange(0, BLOCK_K) < K - off_k
                    x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)

            w = _load_tensor_desc(w_desc, [expt_id, off_k_w, off_n], transpose=W_TRANSPOSE)

            if is_microscaled_format:
                x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
                mx_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)
                if x_format == "fp16" or x_format == "bf16":
                    x_scales: tl.constexpr = None
                else:
                    x_scales = tl.full((BLOCK_M, BLOCK_K // MX_PACK_DIVISOR), 127, dtype=tl.uint8)
                if SWAP_XW:
                    if SWIZZLE_MX:
                        MxPtrs = MxScale + expt_id.to(index_type) * stride_mx_e + offs_mx_outer.to(index_type)[:, None] * stride_mx_n + offs_mx_inner[None, :] + ki * (MX_SCALE_BLOCK_K // 4 * SPLIT_K) * stride_mx_k
                        w_scales = _unswizzle_mx_block(tl.load(MxPtrs))
                    else:
                        MxPtrs = MxScale + expt_id.to(index_type) * stride_mx_e + offs_mx_k.to(index_type)[None, :] * stride_mx_k + offs_w_n.to(index_type)[:, None] * stride_mx_n + ki * MX_SCALE_BLOCK_K * SPLIT_K * stride_mx_k
                        if EVEN_K:
                            if SPLIT_K > 1:
                                w_scales = tl.load(MxPtrs, mask=off_k < K, other=0.0)
                            else:
                                w_scales = tl.load(MxPtrs)
                        else:
                            mask_k = offs_mx_k < tl.cdiv(K - off_k, MX_PACK_DIVISOR)
                            w_scales = tl.load(MxPtrs, mask=mask_k[None, :], other=0.0)

                elif SWIZZLE_MX:
                    w_scales = mx_desc.load([expt_id, off_n // 128, ki * (MX_SCALE_BLOCK_K // 4 * SPLIT_K), 0, 0])
                    w_scales = w_scales.reshape((w_scales.shape[1], w_scales.shape[2] * 32 * 4 * 4))
                    w_scales = _unswizzle_mx_block(w_scales)
                else:
                    w_scales = _load_tensor_desc(mx_desc, [expt_id, off_k_mx, off_n], transpose=MX_TRANSPOSE).T
                if SWAP_XW:
                    acc = tl.dot_scaled(w.T, w_scales, mx_format, x.T, x_scales, x_format, acc=acc, fast_math=True)
                else:
                    acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, mx_format, acc=acc, fast_math=True)
            else:
                if SWAP_XW:
                    acc = tl.dot(w.T, x.T, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                else:
                    acc = tl.dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)

        if INDEPENDENT_EPILOGUE:
            tile_id1 += NUM_SMS
            expt_id1, start_z1, start_m1, eM1, off_m1, off_n1, pid_k1 = _load_tile_attrs(
                tile_id1, num_tiles, grid_m, grid_n, padding_m,
                M, ExptData, ExptHist, ExptOffs,
                BLOCK_M, BLOCK_N, SPLIT_K,
                GROUP_M, XCD_SWIZZLE)
        else:
            tile_id1, expt_id1, start_z1, start_m1, eM1 = tile_id, expt_id, start_z, start_m, eM
            off_m1, off_n1, pid_k1 = off_m, off_n, pid_k

        # Determine output row offsets and mask
        offs_m = off_m1 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M if M is not None else offs_m < eM1
        if HAS_FUSED_SCATTER:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(
                WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            # Later, mask out the acc for computing flexpoint scales.
            MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE

            if USE_SCATTER_TMA and SPLIT_K > 1:
                # Compute the split k offset in number of rows, and add it to offs_y_m.
                # This allows us to write to the correct slice in the output tensor while using
                # a 2D TMA scatter.
                tl.device_assert(stride_y_k // stride_y_m == tl.cdiv(stride_y_k, stride_y_m))
                split_k_row_offs = pid_k1 * (stride_y_k // stride_y_m)
                offs_y_m = tl.where(mask_m, offs_y_m + split_k_row_offs, offs_y_m)
        else:
            offs_y_m = start_m1 + offs_m

            if USE_GATHER_TMA:
                MASK_ACC: tl.constexpr = False
            else:
                # Later, mask out the acc for computing flexpoint scales.
                MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE

        # TMA is faster on Blackwell if a SWAP_XW transpose is not needed, or when we need registers to mask out the acc.
        Y_USE_TMA: tl.constexpr = (MASK_ACC or cuda_capability_geq(10, 0)) and not (DISABLE_Y_TMA or SWAP_XW)

        YBase = Y + start_z1.to(index_type) * stride_y_z + start_m1.to(index_type) * stride_y_m
        if USE_SCATTER_TMA:
            if ExptData is None:  # start_z1 may change; update the descriptor
                y_desc = _update_tensor_desc(y_desc, YBase)
        elif not HAS_FUSED_SCATTER and Y_USE_TMA:
            y_desc = tl.make_tensor_descriptor(
                YBase + pid_k1.to(index_type) * stride_y_k,
                shape=[M if M is not None else eM1, N],
                strides=[stride_y_m, stride_y_n],
                block_shape=[BLOCK_M, EPILOGUE_BLOCK_N],
            )

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
        if PER_BATCH_SCALE:
            w_scale = load_scale(WScale + expt_id1)
        else:
            w_scale = load_scale(WScale)

        if EPILOGUE_SUBTILE:
            accs = tl.split(tl.permute(tl.reshape(acc, (BLOCK_M, 2, EPILOGUE_BLOCK_N)), (0, 2, 1)))
            biases = tl.split(tl.permute(tl.reshape(bias, (2, EPILOGUE_BLOCK_N)), (1, 0)))
        else:
            accs = (acc,)
            biases = (bias,)

        for a_i in tl.static_range(len(accs)):
            acc_tile = accs[a_i]
            acc_tile *= x_scale * w_scale

            if SWAP_XW:
                acc_tile = acc_tile.T
            acc_tile = acc_tile + biases[a_i][None, :] * betas[:, None]
            acc_tile *= gammas[:, None]
            if out_alpha is not None:
                acc_tile *= out_alpha

            if MASK_ACC:
                acc_tile = tl.where(mask_m[:, None], acc_tile, 0.0)

            # Flexpoint
            acc_view = tl.reshape(
                acc_tile, [acc_tile.numel // THREADS_PER_BLOCK, THREADS_PER_BLOCK], can_reorder=True)
            local_absmax = tl.maximum(local_absmax, nan_propagating_absmax_reduce(acc_view, axis=0))
            acc_tile = float_to_flex(
                acc_tile, YExpectedScale,
                None, # ActualScale: local absmax is tracked and updated after the loop
                YChecksumScale,
                None, # mask: acc is manually masked to 0
                Y, FLEXPOINT_SATURATE_INF)
            if EPILOGUE_FN is not None:
                acc_tile = EPILOGUE_FN(acc_tile, *epilogue_fn_args, target_dtype=Y.dtype.element_ty, pid=len(accs)*tile_id1 + a_i)

            if USE_SCATTER_TMA:
                # Convert -1 offsets to INT_MAX. We do this by clearing the leading bit. Note that
                # there shouldn't be any other negative values.
                offs_y_m = (offs_y_m.to(tl.uint32, bitcast=True) & 0x7FFFFFFF).to(tl.int32, bitcast=True)
                y_desc.scatter(acc_tile.to(Y.dtype.element_ty), offs_y_m, off_n1 + a_i * EPILOGUE_BLOCK_N)
            elif not HAS_FUSED_SCATTER and Y_USE_TMA:
                y_desc.store([off_m1, off_n1 + a_i * EPILOGUE_BLOCK_N], acc_tile.to(Y.dtype.element_ty))
            else:
                offs_y_n = off_n1 + a_i * EPILOGUE_BLOCK_N + tl.arange(0, EPILOGUE_BLOCK_N)
                mask_n = offs_y_n < N

                YPtrs = Y + pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_n
                mask = mask_m[:, None] & mask_n[None, :]
                tl.store(YPtrs, acc_tile, mask=mask)


    # Update the flexpoint scales
    if YActualScale is not None:
        tl.atomic_max(YActualScale, compute_scale(local_absmax.to(tl.float32, bitcast=True), Y), sem="relaxed")


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
