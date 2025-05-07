from dataclasses import dataclass
import itertools
import torch
import triton
# utilities
from triton_bench import target_info
from triton_bench.numerics import InFlexData, OutFlexData
from triton_bench.routing import GatherIndx, RoutingData, ScatterIndx
# details
from .matmul_ogs_details._matmul_ogs import _compute_scatter_indx, _matmul_ogs
from .matmul_ogs_details._ptma_matmul_ogs import _ptma_matmul_ogs, get_per_device_per_stream_alloc_fn
from .matmul_ogs_details._finalize_split_k import _finalize_split_k
from .matmul_ogs_details._finalize_scatter import _finalize_scatter
from .matmul_ogs_details.opt_flags import make_opt_flags
from .matmul_ogs_details.metadata import compute_metadata

# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


# ---------------------
# Numerics
# ---------------------

# fmt: off

@dataclass(frozen=True)
class MicroscalingCtx:
    # This interprets the scales as E8M0 tensors
    # Packed fp4s (e2m1) are stored as torch.uint8 tensors.
    # Not used for now, inserted here to make space in the APIs.
    act_scale: torch.Tensor | None = None
    weight_scale: torch.Tensor | None = None

    swizzle_mx: bool = False  # Whether the weight scales are stored in swizzled 5D layout
    actual_weight_scale_shape: tuple | None = None  # Actual weight scales shape, without padding

    def __post_init__(self):
        assert self.act_scale is None, "Activation scale not supported yet"
        if self.weight_scale is None:
            return

        if self.actual_weight_scale_shape is None:
            object.__setattr__(self, "actual_weight_scale_shape", self.weight_scale.shape)

        # Validate the scale tensor data type
        if self.weight_scale.dtype != torch.uint8:
            raise TypeError(f"Weight scale must be uint8. Got {self.weight_scale.dtype}")

        # Validate scale tensor dimensions
        if self.weight_scale.ndim != 3:
            raise ValueError(
                f"Weight scale must be 3D (experts, in_dim // BLOCK_SIZE, out_dim). Got {self.weight_scale.shape}"
            )

    def check_inputs(self, weights: torch.Tensor) -> None:
        if self.weight_scale is None:
            return

        valid_weight_types = {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}
        # Validate weights data type
        if weights.dtype not in valid_weight_types:
            raise TypeError(f"Weights must be one of {valid_weight_types}. Got {weights.dtype}")

        # Validate weights tensor dimensions
        if weights.ndim != 3:
            raise ValueError(f"Weights must be 3D (experts, in_dim, out_dim). Got {weights.shape}")

        # Validate shapes
        weight_scale_shape = self.actual_weight_scale_shape
        if weights.shape[0] != weight_scale_shape[0] or weights.shape[2] != weight_scale_shape[2]:
            raise ValueError(
                f"Weights and scale must have the same number of experts and output dimensions. "
                f"Got weights experts: {weights.shape[0]}, scale experts: {weight_scale_shape[0]}, "
                f"weights out_dim: {weights.shape[2]}, scale out_dim: {weight_scale_shape[2]}"
            )

        k_dim = self.get_packed_tensor_logical_shape(weights)[1]
        rounded_k_dim = (k_dim + 31) // 32 * 32
        block_size = rounded_k_dim // weight_scale_shape[1]
        if block_size != 32:
            raise ValueError(f"Block size must be 32. Got {block_size}")

    def compute_strides(self):
        if self.weight_scale is not None:
            # Check expected properties of the weights.
            if self.swizzle_mx:
                mxE, mxK, mxN = self.weight_scale.shape

                # Compute strides of the 5D swizzled tensor.
                swizzled_shape = (mxE, mxN // 128, mxK // 4, 32, 4, 4)
                s5 = 1
                s4 = swizzled_shape[5] * s5       # 4 * 1 = 4
                s3 = swizzled_shape[4] * s4       # 32 * 4 = 128
                s2 = swizzled_shape[3] * s3       # 4 * 128 = 512
                s1 = swizzled_shape[2] * s2       # (mxK//4) * 512
                s0 = swizzled_shape[1] * s1       # (mxN//128) * ((mxK//4)*512)
                mx_scale_stride_e, mx_scale_stride_n, mx_scale_stride_k = s0, s1, s2
            else:
                mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n = self.weight_scale.stride()
        else:
            mx_scale_stride_e = mx_scale_stride_k = mx_scale_stride_n = 0
        return mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n


    def get_packed_tensor_logical_shape(self, tensor: torch.Tensor):
        k_dim = tensor.shape[1]
        if tensor.dtype == torch.uint8:
            # Assume 2 fp4s packed into a byte
            k_dim *= 2
        return tensor.shape[0], k_dim, tensor.shape[2]

@dataclass(frozen=True)
class FlexCtx:
    lhs_data: InFlexData = InFlexData()
    rhs_data: InFlexData = InFlexData()
    out_data: OutFlexData = OutFlexData()

@dataclass
class PrecisionConfig:
    max_num_imprecise_acc: int = None
    allow_tf32: bool = True
    flex_ctx: FlexCtx = FlexCtx()
    acc_scale: int = 1.0
    flexpoint_saturate_inf: bool = False
    report_quantization_err_fn: callable = None

    mx_ctx: MicroscalingCtx = MicroscalingCtx()
    out_dtype: torch.dtype = None
    enforce_bitwise_invariance: bool = False

    def __post_init__(self):
        empty_flex = FlexCtx()
        assert self.flex_ctx.rhs_data == empty_flex.rhs_data or self.mx_ctx.weight_scale is None, "flex and mx_ctx cannot be used together"

def mx_can_use_tma(mx_ctx: MicroscalingCtx):
    mx_scale_stride_e, mx_scale_stride_n, mx_scale_stride_k = mx_ctx.compute_strides()
    if mx_scale_stride_e * mx_ctx.weight_scale.element_size() % 16 != 0:
        return False

    if mx_ctx.swizzle_mx:
        # CHeck stride in bytes are multiples of 16.
        return mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0
    else:
        # Check MX is either transposed or non-transposed, and with required stride.
        return (
            (mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k == 1) or
            (mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_n == 1)
        )

def can_use_persistent_tma(x, w, gather_indx, precision_config):
    mx_ctx = precision_config.mx_ctx
    return (
        # TMA requires CUDA 9.0, last dim contiguous, and multiple of 16-byte strides otherwise.
        target_info.cuda_capability_geq(9, 0) and
        (True if gather_indx is not None else
            # Check strides of X.
            x.stride(1) * x.element_size() % 16 == 0 and x.stride(2) == 1
        ) and (
            # Check W is either transposed or non-transposed, and with required stride.
            (w.stride(1) * w.element_size() % 16 == 0 and w.stride(2) == 1) or
            (w.stride(2) * w.element_size() % 16 == 0 and w.stride(1) == 1)
        ) and (
            mx_ctx.weight_scale is None or mx_can_use_tma(mx_ctx)
        ) and (
            # MFXP4 tma requires 128 elements on the inner dim.
            # MFXP4 is represented as packed uint8.
            w.dtype != torch.uint8 or w.shape[-1] % 128 == 0
        )
        # compiler crash ?
        and (x.dtype.itemsize <= 1 or w.dtype != torch.uint8)
    )

def can_use_fused_scatter(scatter_indx):
    return scatter_indx is not None

# ---------------------
# Preprocessing
# ---------------------

@dataclass(frozen=True)
class PreprocessingFeatures:
    w_want_n_major: bool
    w_want_k_major: bool
    swap_xw: bool

    def __post_init__(self):
        assert not (self.w_want_k_major and self.w_want_n_major), "Cannot have both K-major and N-major"

def init_preprocessing_features(w, precision_config, opt_flags):
    mx_ctx = precision_config.mx_ctx
    swap_xw = False  # Whether or not to swap X and W operands to the tl.dot
    w_want_k_major = False
    w_want_n_major = False
    if not target_info.cuda_capability_geq(10, 0):
        # Hopper transpose. Reduction dimension must be contiguous.
        if w.stride(1) != 1 and w.dtype.itemsize == 1:
            w_want_k_major = True

    if target_info.cuda_capability_geq(10, 0):
        swap_xw = mx_ctx.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
        if swap_xw:
            w_want_k_major = True
        # fp4 padded mode requires the contiguous dim size to be a multiple of 64 bytes. If it is K-major and does not
        # meet the requirement, make the tensor N-major instead.
        # But, don't do this if we're going to swap X and W in which case we would transpose W again.
        if w.stride(1) == 1 and w.dtype == torch.uint8 and w.shape[1] % 64 != 0 and not swap_xw:
            w_want_n_major = True
    if not w_want_k_major and not w_want_n_major:
        w_want_k_major = True
    return PreprocessingFeatures(w_want_n_major, w_want_k_major, swap_xw)

def apply_preprocessing_features(x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features):
    has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
    if has_fused_scatter_scratchpad:
        M = scatter_indx.src_indx.shape[0]
        writeback_idxs = torch.empty((M,), dtype=torch.int32, device=x.device)
        writeback_size = writeback_idxs.shape[0]
        BLOCK_M=256
        _compute_scatter_indx[(triton.cdiv(M, BLOCK_M),)](
            writeback_idxs,
            scatter_indx.dst_indx,
            scatter_indx.src_indx,
            M // routing_data.n_expts_act,
            M,
            BLOCK_M=BLOCK_M,
            N_EXPTS_ACT=routing_data.n_expts_act,
        )
    elif scatter_indx is not None and routing_data.n_expts_act == 1:
        writeback_idxs = scatter_indx.dst_indx
        writeback_size = scatter_indx.dst_indx.shape[0]
    else:
        writeback_idxs, writeback_size = None, None
    # some transposition variants aren't supported
    # TODO: this is extremely expensive and we should find
    # a way to surface this to the user
    if preprocessing_features.w_want_n_major:
        w = w.contiguous()
    elif preprocessing_features.w_want_k_major:
        w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
    # preprocess routing information and ptr lookup table
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    # compute expt_data
    expt_data = compute_metadata(routing_data, M, opt_flags.block_m)
    return x, w, preprocessing_features.swap_xw, writeback_idxs, writeback_size, expt_data

# ---------------------
# Postprocessing
# ---------------------

@dataclass(frozen=True)
class PostprocessingFeatures:
    finalize_splitk: bool
    finalize_scatter: bool

    def __post_init__(self):
        assert not (self.finalize_splitk and self.finalize_scatter)

def init_postprocessing_features(routing_data, scatter_indx, opt_flags):
    finalize_scatter = scatter_indx is not None and routing_data.n_expts_act > 1
    # TODO: there should be an assert somewhere!
    finalize_splitk = opt_flags.split_k > 1 and not finalize_scatter
    return PostprocessingFeatures(finalize_splitk=finalize_splitk,
                                  finalize_scatter=finalize_scatter)

def apply_postprocessing_features(scatter_indx, opt_flags, expt_offs, num_indx, precision_config, routing_data,
                           postprocess_features, memory):
    out = memory["output"]
    flex_ctx = precision_config.flex_ctx
    # finalize split-k
    if postprocess_features.finalize_splitk:
        inp = memory["scratchpad"]["matmul"]
        out_splitk = memory["output"]
        out_splitk_flex = precision_config.flex_ctx.out_data
        assert out_splitk.stride(3) == 1
        flattened_M = inp.shape[1] * inp.shape[2]
        N = inp.shape[3]
        grid = (flattened_M, triton.cdiv(N, opt_flags.block_n))
        _finalize_split_k[grid](
            inp, inp.stride(0), inp.stride(2),
            flex_ctx.out_data.reinterpret(out_splitk), out_splitk.stride(2),
            *out_splitk_flex,
            flattened_M, N, opt_flags.split_k,
            None if expt_offs is None else expt_offs[-1],
            num_indx,
            1,
            opt_flags.block_n,
            precision_config.flexpoint_saturate_inf,
        )
        out = out_splitk
    # finalize scatter
    # batched mode not supported.
    if postprocess_features.finalize_scatter:
        has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
        if has_fused_scatter_scratchpad:
            inp = memory["output"]
        else:
            inp = memory["scratchpad"]["matmul"]
        n_final_rows = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act
        inp_flex = OutFlexData() if inp.dtype == torch.float32 else precision_config.flex_ctx.out_data
        out_scatter = memory["output"]
        out_scatter_flex = precision_config.flex_ctx.out_data
        assert inp.shape[1] == 1
        if target_info.is_hip():
            num_warps = 2
            BLOCK_N = 2048
            warps_per_sm = 32
        else:
            num_warps = 16
            BLOCK_N = 4096
            warps_per_sm = 128
        num_pid = target_info.num_sms() * (warps_per_sm // num_warps)
        N = inp.shape[3]
        M = n_final_rows
        # assert M == out_scatter.shape[1], f"{M}, {out_scatter.shape}"
        N_BLOCKS = triton.cdiv(N, BLOCK_N)
        M_BLOCKS = min(M, max(1, triton.cdiv(num_pid, N_BLOCKS)))
        grid = (M_BLOCKS, N_BLOCKS)
        _finalize_scatter[grid](
            flex_ctx.out_data.reinterpret(out_scatter),
            *out_scatter_flex,
            flex_ctx.out_data.reinterpret(inp), inp.stride(0), inp.stride(2),
            inp_flex.expected_scale,
            scatter_indx.src_indx,
            inp.shape[0], M, N,
            None if expt_offs is None else expt_offs[-1],
            EXPT_PER_TOK=routing_data.n_expts_act,
            BLOCK_N=BLOCK_N,
            M_BLOCKS=M_BLOCKS,
            num_warps=num_warps,
            flexpoint_saturate_inf=precision_config.flexpoint_saturate_inf,
            HAS_FUSED_SCRATCHPAD=has_fused_scatter_scratchpad,
        )
        out = out_scatter
        # trim unnecessary part of output
        if has_fused_scatter_scratchpad:
            # Discard scratchpad part.
            # This still gives a contiguous tensor, because shape[0] > 1 only when
            # batch mode is enabled, in which case this is a no-op (there's no scratchpad).
            out = out[:, :, :n_final_rows, :]
    return out


# ---------------------
# Allocation
# ---------------------

@dataclass
class MatmulAllocation:
    device: str
    output: tuple[tuple[int], torch.dtype]
    scratchpads: dict[str, tuple]


def init_allocation(x, w, precision_config, routing_data, gather_indx, scatter_indx, opt_flags,
                    preprocessing_features, postprocessing_features):
    # ---- output ------
    N = precision_config.mx_ctx.get_packed_tensor_logical_shape(w)[-1]
    # by default - M is number of rows in the activations
    M = x.shape[1]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.src_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    elif opt_flags.fused_scatter:
        # we need the scratchpad and the output to be contiguous in memory
        Mc = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
        y_rows = M + Mc
    else:
        Mc = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
        y_rows = Mc
    y_shape = (x.shape[0], y_rows, N)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (y_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    # if we need either standalone scatter or split-k, the matmul output will need post-processing
    if postprocessing_features.finalize_splitk or (postprocessing_features.finalize_scatter and not opt_flags.fused_scatter):
        dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
        scratchpad["matmul"] = ((opt_flags.split_k, x.shape[0], M, N), dtype)
    return MatmulAllocation(x.device, output, scratchpad)

def apply_allocation(allocation: MatmulAllocation, output):
    ret = dict()
    if output is None:
        output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
    else:
        assert output.shape == allocation.output[0]
    ret["output"] = output[None, :, :]
    ret["scratchpad"] = {
        k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            for k, v in allocation.scratchpads.items()
    }
    return ret

# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------

def matmul_ogs(x, w, bias,
               routing_data: RoutingData | None = None,
               gather_indx: GatherIndx | None = None,
               scatter_indx: ScatterIndx | None = None,
               precision_config: PrecisionConfig | None = None,
               betas: torch.Tensor | None = None,
               gammas: torch.Tensor | None = None,
               out_alpha: float | None = None,
               y: torch.Tensor | None = None,
               ):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """

    is_input_batched = x.ndim == 3
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if precision_config is None:
        precision_config = PrecisionConfig()
    if w.ndim == 2:
        w = w.view(1, w.shape[-2], w.shape[-1])
    if x.ndim == 2:
        x = x.view(1, x.shape[-2], x.shape[-1])
    assert w.ndim == 3
    assert x.ndim == 3
    # unpack scales
    mx_ctx = precision_config.mx_ctx
    # determine shapes
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    batch_size = w.shape[0] if routing_data.expt_hist is None else 1
    n_expts_tot, K, N = mx_ctx.get_packed_tensor_logical_shape(w)
    mx_ctx.check_inputs(w)
    mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n = mx_ctx.compute_strides()
    # compute optimization flags
    out_dtype = precision_config.out_dtype or x.dtype
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        M, N, K, routing_data,
        can_use_persistent_tma(x, w, gather_indx, precision_config),
        can_use_fused_scatter(scatter_indx),
    )
    # compute grid size
    if not is_input_batched:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    else:
        grid_m = triton.cdiv(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    assert n_expts_tot == routing_data.n_expts_tot
    assert grid_m > 0
    assert x.dtype == w.dtype or mx_ctx.weight_scale is not None
    # determine necessary pre/post processing
    preprocessing_features = init_preprocessing_features(w, precision_config, opt_flags)
    postprocessing_features = init_postprocessing_features(routing_data, scatter_indx, opt_flags)
    # allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, routing_data, gather_indx, scatter_indx, opt_flags,
                                 preprocessing_features, postprocessing_features)
    memory = apply_allocation(allocation, y)
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    # Intermediate tensors and postprocess kernels for each situation
    out0, out0_flex = memory["output"], precision_config.flex_ctx.out_data
    if postprocessing_features.finalize_scatter or postprocessing_features.finalize_splitk:
        if opt_flags.fused_scatter:
            out0 = memory["output"]
        else:
            out0 = memory["scratchpad"]["matmul"]
        out0_flex = OutFlexData() if out0.dtype == torch.float32 else precision_config.flex_ctx.out_data
    # pre-processing
    x, w, swap_xw, writeback_idxs, writeback_size, expt_data  = apply_preprocessing_features(
        x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features
    )
    if expt_data.buffer is not None:
        assert expt_data.buffer.shape[0] == 3*n_expts_tot + 2 + grid_m, \
            f"invalid expt_data, {expt_data.buffer.shape}, {n_expts_tot=}, {grid_m=}"
    # matrix multiplication
    n_cta = batch_size * grid_m * grid_n * opt_flags.split_k
    n_cta = min(target_info.num_sms(), n_cta) if opt_flags.is_persistent else n_cta
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]
    (_ptma_matmul_ogs if opt_flags.is_persistent else _matmul_ogs)[(n_cta,)](
                   flex.out_data.reinterpret(memory["output"]),
                   flex.out_data.reinterpret(out0), *out0.stride(),
                   *out0_flex,
                   flex.lhs_data.reinterpret(x), x.stride(0), x.stride(1), x.stride(2),
                   flex.lhs_data.scale,
                   flex.rhs_data.reinterpret(w), w.stride(0), w.stride(1), w.stride(2), w.stride(2) != 1,
                   flex.rhs_data.scale,
                   mx_ctx.weight_scale, mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n, mx_scale_stride_n != 1,
                   bias, bias_stride,
                   x.shape[1],
                   x.shape[1] if routing_data.expt_hist is None else None,
                   N, K,
                   betas, gammas,
                   None if gather_indx is None else gather_indx.src_indx,
                   None if scatter_indx is None else scatter_indx.src_indx,
                   num_indx,
                   writeback_idxs, writeback_size,
                   expt_data.hist, expt_data.offs, expt_data.offs_sum, expt_data.blocks,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   routing_data.n_expts_tot, routing_data.n_expts_act,
                   precision_config.max_num_imprecise_acc,
                   precision_config.allow_tf32,
                   precision_config.flexpoint_saturate_inf,
                   flex.rhs_data.is_per_batch,
                   opt_flags.block_m,
                   opt_flags.block_n,
                   opt_flags.block_k,
                   opt_flags.group_m,
                   XCD_SWIZZLE=opt_flags.xcd_swizzle,
                   SWIZZLE_MX=mx_ctx.swizzle_mx,
                   EPILOGUE_SUBTILE=opt_flags.epilogue_subtile,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=K % opt_flags.block_k == 0,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=routing_data.expected_tokens_per_expt,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(x, w, out0),
                   DISABLE_Y_TMA=out0.stride(-2) * out0.dtype.itemsize % 16 != 0,
                   SWAP_XW=swap_xw,
                   NUM_SMS = n_cta,
                   **opt_flags.target_kernel_kwargs)
    # post-processing
    out = apply_postprocessing_features(scatter_indx, opt_flags, expt_data.offs,
                                num_indx, precision_config, routing_data,
                                postprocessing_features, memory)

    # remove split-k
    out = out.squeeze(0)
    if not is_input_batched:
        out = out.view(out.shape[-2], out.shape[-1])
    return out


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------

def matmul_ogs_torch(x, w, bias,
                 routing_data: RoutingData = None,
                 gather_indx: GatherIndx = None,
                 scatter_indx: ScatterIndx = None,
                 precision_config: PrecisionConfig = None,
                 betas = None,
                 gammas = None,
                 round_x = None, round_y = None,
                 ):
    is_input_batched = x.ndim == 3
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if round_x is None:
        round_x = lambda x: x
    if round_y is None:
        round_y = lambda x: x
    if w.ndim == 2:
        w = w.view(1, w.shape[0], w.shape[1])
    if x.ndim == 2:
        x = x.view(1, x.shape[0], x.shape[1])
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1 and not is_input_batched:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[1]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[1] if gather_indx is None else gather_indx.dst_indx.shape[0]
    y = torch.zeros((x.shape[0], n_rows, w.shape[-1]), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx.src_indx[lo:hi] // n_expts_act
        batch = i if is_input_batched else 0
        out = torch.matmul(round_x(x[batch, idx, :], torch.arange(lo, hi, device="cuda")).float(),
                           w[i, :, :].float())
        if bias is not None:
            out += bias[i, :] if betas is None else bias[i, :] * betas[lo:hi, None]
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[batch, lo:hi, :] = round_y(out)
    if not is_input_batched:
        y = y.view(y.shape[1], y.shape[2])
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    for i, (lo, hi) in enumerate(offs):
        dst_idx = scatter_indx.dst_indx[lo:hi] // n_expts_act
        msk = dst_idx != -1
        out[dst_idx[msk], :] += y[lo:hi, :][msk, :].float()
    return out
