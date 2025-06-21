from dataclasses import dataclass
import itertools
import sys
import torch
import triton
# utilities
from triton_kernels import target_info
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.routing import ExptData, GatherIndx, RoutingData, ScatterIndx
from triton.tools.tensor_descriptor import TensorDescriptor
# details
from .matmul_ogs_details._matmul_ogs import _compute_writeback_idx
from .matmul_ogs_details._matmul_ogs import _matmul_ogs
from .matmul_ogs_details._p_matmul_ogs import _p_matmul_ogs, get_per_device_per_stream_alloc_fn
from .matmul_ogs_details._finalize_matmul import _finalize_matmul
from .matmul_ogs_details.opt_flags import make_opt_flags, OptFlags, update_opt_flags_constraints
from .numerics_details.mxfp import SwizzlingType
from .specialize import specialize
from .datastruct import Tensor
from typing import Tuple, Optional


@dataclass
class FnSpecs:
    name: str
    fn: "triton.runtime.jit.JITFunction"
    fn_arg_names: tuple[str]
    fn_arg_do_not_specialize: tuple[str] = tuple()

    @staticmethod
    def default():
        return FnSpecs("dflt", None, tuple())


@dataclass
class FusedActivation:
    specs: FnSpecs
    fn_args: tuple[object]
    reduction_n: int


@dataclass
class Epilogue:
    specs: FnSpecs
    fn_arg_values_matmul: tuple[object]
    fn_arg_values_finalize: tuple[object]
    effective_itemsize: float | None = None


EpilogueSpecs = FnSpecs  # TODO: remove this alias when callers are updated

_kernels = dict()


def get_kernels(epilogue: FnSpecs = FnSpecs.default(), fused_activation: FnSpecs = FnSpecs.default()):
    global _kernels
    key = (fused_activation.name, epilogue.name)
    if key in _kernels:
        return _kernels[key]
    spec_constants = {
        "ACTIVATION_FN": fused_activation.fn,
        "EPILOGUE_FN": epilogue.fn,
    }
    spec_tuples = {
        "activation_fn_args": fused_activation.fn_arg_names,
        "epilogue_fn_args": epilogue.fn_arg_names,
    }
    do_not_specialize = fused_activation.fn_arg_do_not_specialize + epilogue.fn_arg_do_not_specialize
    import types

    module = types.ModuleType(f"matmul_ogs_{'_'.join(key)}")
    sys.modules[module.__name__] = module
    module._finalize_matmul = specialize(_finalize_matmul, module, spec_constants, spec_tuples,
                                         do_not_specialize=do_not_specialize)
    module._matmul_ogs = specialize(_matmul_ogs, module, spec_constants, spec_tuples,
                                    do_not_specialize=do_not_specialize)
    module._p_matmul_ogs = specialize(_p_matmul_ogs, module, spec_constants, spec_tuples,
                                      do_not_specialize=do_not_specialize)
    _kernels[key] = module
    return module


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


class TensorDescriptorBuilder:
    """Builder for creating different types of tensor descriptors"""

    @staticmethod
    def create_block_scale_descriptor(mx_tensor: torch.Tensor, block_k: int, block_n: int, B: int, K: int, N: int,
                                      mx_scale_stride_k: int, mx_scale_stride_n: int, swizzle_mx: bool,
                                      transpose: Optional[bool]) -> TensorDescriptor:
        """Create a tensor descriptor for block scale factors"""
        assert swizzle_mx, "only support swizzled block scales"
        assert block_n >= 128
        assert transpose is None
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_k // MX_PACK_DIVISOR
        PackedK = triton.cdiv(K, MX_PACK_DIVISOR)
        num_expt_x_ncol = B * triton.cdiv(N, 128)
        shape = [1, num_expt_x_ncol, triton.cdiv(PackedK, 4), 2, 256]
        strides = [num_expt_x_ncol * mx_scale_stride_n, mx_scale_stride_n, mx_scale_stride_k, 256, 1]
        block_shape = [1, block_n // 128, MX_SCALE_BLOCK_K // 4, 2, 256]
        return TensorDescriptor(base=mx_tensor, shape=shape, strides=strides, block_shape=block_shape)

    @staticmethod
    def create_descriptor(tensor: torch.Tensor, block_shape: list[int], transpose=False) -> TensorDescriptor:
        """Create a tensor descriptor for matrix X via TMA"""
        assert tensor.ndim in [2, 3], "TMA descriptor builder expects 2D or 3D input"
        assert tensor.ndim == len(block_shape)
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            tensor = tensor.permute(0, 2, 1)
        if isinstance(tensor, (Tensor)):
            tensor = tensor.handle
        # FIXME: incorrect in the general case
        PACK_DIVISOR = 2 if tensor.dtype == torch.uint8 else 1
        block_shape[-1] = block_shape[-1] // PACK_DIVISOR
        return TensorDescriptor.from_tensor(tensor, block_shape=block_shape)


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

    swizzle_value: SwizzlingType | None = (
        None  # Whether the weight values are stored in swizzled layout
    )
    swizzle_scale: SwizzlingType | None = (
        None  # Whether the weight scales are stored in swizzled layout
    )
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


def element_bitwidth(tensor):
    return tensor.element_bitwidth if isinstance(tensor, Tensor) else tensor.element_size()*8

def none_or_tma_compatible(x):
    if x is None:
        return True
    if not target_info.cuda_capability_geq(9, 0):
        return False
    if x.ndim != 3:
        return False
    strides = list(x.stride())
    stride_div = 128 // element_bitwidth(x)
    try:
        major_dim = strides.index(1)
    except ValueError:
        major_dim = -1
        # return False
    compliant = [x.stride(i)*x.element_size() % stride_div == 0 for i in range(x.ndim) if i != major_dim]
    return all(compliant)

# ---------------------
# Preprocessing
# ---------------------

@dataclass(frozen=True)
class PreprocessingFeatures:
    swap_xw: bool


def init_preprocessing_features(w, precision_config, opt_flags):
    mx_ctx = precision_config.mx_ctx
    swap_xw = False  # Whether or not to swap X and W operands to the tl.dot
    if target_info.cuda_capability_geq(10, 0):
        swap_xw = mx_ctx.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
    return PreprocessingFeatures(swap_xw)

def apply_preprocessing_features(x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features):
    has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
    if has_fused_scatter_scratchpad:
        M = scatter_indx.src_indx.shape[0]
        writeback_idxs = torch.zeros((M,), dtype=torch.int32, device=x.device)
        writeback_size = writeback_idxs.shape[0]
        finalize_scatter_idxs = torch.zeros((M // routing_data.n_expts_act + M + 1,), dtype=torch.int32, device=x.device)
        BLOCK_M=256
        _compute_writeback_idx[(triton.cdiv(M, BLOCK_M),)](
            writeback_idxs,
            finalize_scatter_idxs,
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
        finalize_scatter_idxs = None
    else:
        writeback_idxs, writeback_size, finalize_scatter_idxs = None, None, None
    # preprocess routing information and ptr lookup table
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    return x, w, writeback_idxs, writeback_size, finalize_scatter_idxs


# ---------------------
# Postprocessing
# ---------------------


@dataclass(frozen=True)
class PostprocessingFeatures:
    finalize: bool

def init_postprocessing_features(routing_data, scatter_indx, opt_flags):
    finalize = (scatter_indx is not None and routing_data.n_expts_act > 1) or opt_flags.split_k > 1
    return PostprocessingFeatures(finalize)

def apply_postprocessing_features(scatter_indx, finalize_scatter_idxs, opt_flags, expt_offs, num_indx, precision_config, routing_data,
                                  postprocess_features, memory, fused_activation, epilogue):
    out = memory["output"]
    flex_ctx = precision_config.flex_ctx
    if postprocess_features.finalize:
        has_fused_scatter_scratchpad = opt_flags.fused_scatter and routing_data.n_expts_act > 1
        if has_fused_scatter_scratchpad:
            inp = memory["output"]
        else:
            inp = memory["scratchpad"]["matmul"]
        if scatter_indx is not None:
            assert inp.shape[1] == 1, "batched finalize scatter not supported"
            n_final_rows = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act
            scatter_src_indx = scatter_indx.src_indx
            EXPT_PER_TOK = routing_data.n_expts_act
            num_rows = None
        else:
            n_final_rows = inp.shape[1] * inp.shape[2]
            scatter_src_indx = None
            EXPT_PER_TOK = 1
            num_rows = num_indx or (None if expt_offs is None else expt_offs[-1])

        if inp.dtype == torch.float32:
            inp_flex = OutFlexData()
        else:
            inp_flex = precision_config.flex_ctx.out_data

        out_scatter = memory["output"]
        out_scatter_flex = precision_config.flex_ctx.out_data

        N = inp.shape[3]
        M = n_final_rows
        warps_per_sm = 32 if target_info.is_hip() else 128

        def compute_grid(BLOCK_N, num_warps):
            num_pid = target_info.num_sms() * (warps_per_sm // num_warps)
            if M < num_pid or target_info.is_hip():
                grid_n = triton.cdiv(N, BLOCK_N)
                grid_m = min(M, max(1, triton.cdiv(num_pid, grid_n)))
            else:
                grid_m = min(M, num_pid)
                grid_n = 1
            return (grid_m, grid_n)

        if inp.dtype.itemsize == 1:
            candidates = [(1024, 1)]
        else:
            if target_info.is_hip():
                candidates = [(4096 // inp.dtype.itemsize, 2)]
            else:
                if inp.dtype.itemsize == 2:
                    candidates = [
                        (4096 // inp.dtype.itemsize, 4),
                        (1024 // inp.dtype.itemsize, 1),
                    ]
                else:
                    candidates = [
                        (2048 // inp.dtype.itemsize, 4),
                        (1024 // inp.dtype.itemsize, 1),
                    ]
        if precision_config.enforce_bitwise_invariance:
            candidates = [candidates[0]]

        # sort by smallest grid_n so we share compute across a row
        grid, (BLOCK_N, num_warps) = sorted([(compute_grid(*c), c) for c in candidates], key=lambda x: x[0][1])[0]
        STAGES = 1 if num_warps == 1 else min(triton.cdiv(triton.cdiv(N, BLOCK_N), grid[1]), 5)

        kernels = get_kernels(epilogue.specs, fused_activation.specs)
        kernels._finalize_matmul[grid](
            flex_ctx.out_data.reinterpret(out_scatter),
            *out_scatter_flex,
            flex_ctx.out_data.reinterpret(inp), inp.stride(0), inp.stride(2),
            inp_flex.expected_scale,
            scatter_src_indx, finalize_scatter_idxs,
            inp.shape[0], M, N, num_rows,
            *fused_activation.fn_args, fused_activation.reduction_n,
            *epilogue.fn_arg_values_finalize,
            EXPT_PER_TOK=EXPT_PER_TOK,
            BLOCK_N=BLOCK_N,
            STAGES=STAGES,
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

def init_allocation(x, w, precision_config, fused_activation, routing_data, gather_indx, scatter_indx, opt_flags,
                    preprocessing_features, postprocessing_features):
    # ---- output ------
    N = w.shape[-1]
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
    y_shape = (x.shape[0], y_rows, N // fused_activation.reduction_n)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (y_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    # if we need either standalone scatter or split-k, the matmul output will need post-processing
    if postprocessing_features.finalize and (opt_flags.split_k > 1 or not opt_flags.fused_scatter):
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

def _create_tma_descriptors(
    x: torch.Tensor,
    w: torch.Tensor,
    mx_tensor: Optional[torch.Tensor],
    routing_data: RoutingData,
    mx_ctx: MicroscalingCtx,
    expt_data: ExptData,
    opt_flags: OptFlags,
    B: int,
    K: int,
    N: int,
    mx_scale_stride_k: int,
    mx_scale_stride_n: int,
    HAS_GATHER: bool,
) -> Tuple[bool, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Create and cache TMA descriptors for tensors."""

    x_tensor_or_desc, mx_desc_and_transpose = x, (None, False)
    x_has_tma = (not HAS_GATHER) or (HAS_GATHER and target_info.has_tma_gather())
    if x_has_tma:
        block_m = [1] if HAS_GATHER else [opt_flags.block_m]
        x_block_shape = [1] * (x.ndim - 2) + block_m + [opt_flags.block_k]
        x_tensor_or_desc = TensorDescriptorBuilder.create_descriptor(x, x_block_shape)

    w_transpose = w.stride(2) != 1
    w_block_shape = [1, opt_flags.block_k, opt_flags.block_n]
    w_desc = TensorDescriptorBuilder.create_descriptor(w, w_block_shape, transpose=w_transpose)
    w_desc_and_transpose = (w_desc, w_transpose)

    is_microscaled_format = mx_ctx.weight_scale is not None and w.dtype == torch.uint8
    if is_microscaled_format:
        # Pad the inner shape to 128 for mxfp4 weights; TMA requires this when the compiler uses
        # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B.
        # This technically makes the shape masking incorrect, but it's fine because:
        #  - When the N dim is padded, the scales will be masked to 0.
        #  - When the K dim is padded, the activations we perform tl.dot with will be masked to 0.
        #    Note: the scales can't be relied on for zeroing in this case, because they apply to groups
        #    of 32 elements in the K dimension.
        pad = 128
        dim_to_pad = -1
        old_size = w_desc.shape[dim_to_pad]
        padded_size = triton.cdiv(old_size, pad) * pad
        if padded_size != old_size:
            w_desc.shape = list(w_desc.shape)
            w_desc.shape[dim_to_pad] = padded_size

    if mx_tensor is not None:
        mx_transpose = mx_scale_stride_n != 1 if mx_ctx.swizzle_scale is None else None
        mx_desc = TensorDescriptorBuilder.create_block_scale_descriptor(
                mx_tensor, opt_flags.block_k, opt_flags.block_n,
                routing_data.n_expts_tot if expt_data is not None and len(expt_data.block_pid_map) > 0 else B, K, N,
                mx_scale_stride_k, mx_scale_stride_n, mx_ctx.swizzle_scale, mx_transpose
            )
        mx_desc_and_transpose = (mx_desc, mx_transpose)

    return x_tensor_or_desc, w_desc_and_transpose, mx_desc_and_transpose

def matmul_ogs_set_idle_sms(num_idle_sms):
    """
    persistent kernels will leave `num_idle_sms` idle
    """
    update_opt_flags_constraints({"idle_sms": num_idle_sms})

def matmul_ogs(x, w, bias,
               routing_data: RoutingData | None = None,
               gather_indx: GatherIndx | None = None,
               scatter_indx: ScatterIndx | None = None,
               precision_config: PrecisionConfig | None = None,
               betas: torch.Tensor | None = None,
               gammas: torch.Tensor | None = None,
               out_alpha: float | None = None,
               y: torch.Tensor | None = None,
               fused_activation: FusedActivation | None = None,
               epilogue: Epilogue | None = None,
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
    # canonicalize inputs
    if precision_config is None:
        precision_config = PrecisionConfig()
    if fused_activation is None:
        fused_activation = FusedActivation(FnSpecs.default(), tuple(), 1)
    if epilogue is None:
        epilogue = Epilogue(FnSpecs.default(), tuple(), tuple(), False)
    if w.ndim == 2:
        w = w.view(1, w.shape[-2], w.shape[-1])
    if x.ndim == 2:
        x = x.view(1, x.shape[-2], x.shape[-1])
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    assert w.ndim == 3
    assert x.ndim == 3
    assert w.shape[0] == routing_data.n_expts_tot
    # unpack scales
    mx_ctx = precision_config.mx_ctx
    has_mx = mx_ctx.weight_scale is not None
    is_hopper_fp8 = not target_info.cuda_capability_geq(10, 0) and w.dtype.itemsize == 1
    if has_mx: assert w.stride(1) == 1, "`w` must be column-major when it has data-type mxfp"
    if is_hopper_fp8: assert w.stride(1) == 1, "`w` must be column-major when it has data-type FP8 on capability < 10"
    w_scale = None if mx_ctx.weight_scale is None else Tensor(mx_ctx.weight_scale, swizzle_mode=mx_ctx.swizzle_scale)
    w = w if w_scale is None else Tensor(w, swizzle_mode=mx_ctx.swizzle_value)
    # determine shapes
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    batch_size = w.shape[0] if routing_data.expt_hist is None else 1
    _, K, N = w.shape
    mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n = w_scale.strides if has_mx else (0,0,0)
    # compute optimization flags
    out_dtype = precision_config.out_dtype or x.dtype
    can_use_tma = none_or_tma_compatible(x) and none_or_tma_compatible(w) and none_or_tma_compatible(w_scale)
    can_use_fused_scatter = scatter_indx is not None and fused_activation.specs.fn is None
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        M, N, K, routing_data, can_use_tma, can_use_fused_scatter, epilogue.effective_itemsize,
    )
    # determine necessary pre/post processing
    preprocessing_features = init_preprocessing_features(w, precision_config, opt_flags)
    postprocessing_features = init_postprocessing_features(routing_data, scatter_indx, opt_flags)
    # allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, fused_activation, routing_data, gather_indx, scatter_indx, opt_flags,
                                 preprocessing_features, postprocessing_features)
    memory = apply_allocation(allocation, y)
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    # Intermediate tensors and postprocess kernels for each situation
    out0, out0_flex = memory["output"], precision_config.flex_ctx.out_data
    fused_postprocess_activation = FusedActivation(FnSpecs.default(), tuple(), 1)
    if postprocessing_features.finalize:
        if opt_flags.fused_scatter:
            out0 = memory["output"]
        else:
            out0 = memory["scratchpad"]["matmul"]
        out0_flex = OutFlexData() if out0.dtype == torch.float32 else precision_config.flex_ctx.out_data

        fused_activation, fused_postprocess_activation = fused_postprocess_activation, fused_activation
    # pre-processing
    x, w, writeback_idxs, writeback_size, finalize_scatter_idxs = apply_preprocessing_features(
        x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features
    )
    # matrix multiplication


    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]

    expt_data = routing_data.expt_data
    block_m = opt_flags.block_m
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map[block_m]

    grid_m = triton.cdiv(M, opt_flags.block_m)
    if expt_block_pid_map is not None:
        grid_m = expt_block_pid_map.numel()
    grid_n = triton.cdiv(N, opt_flags.block_n)
    max_grid = batch_size * grid_m * grid_n * opt_flags.split_k
    grid = min(target_info.num_sms() - opt_flags.idle_sms, max_grid) if opt_flags.is_persistent else max_grid

    x = flex.lhs_data.reinterpret(x)
    w = flex.rhs_data.reinterpret(w)

    if opt_flags.is_persistent:
        x_tensor, w_tensor_and_transpose, mx_tensor_and_tranpose = _create_tma_descriptors(
            x=x, w=w, mx_tensor=mx_ctx.weight_scale,
            routing_data=routing_data,
            mx_ctx=mx_ctx,
            expt_data=expt_data,
            opt_flags=opt_flags,
            B=batch_size,
            K=K,
            N=N,
            mx_scale_stride_k=mx_scale_stride_k,
            mx_scale_stride_n=mx_scale_stride_n,
            HAS_GATHER=gather_indx is not None,
        )
        w_tensor, w_tma_transpose = w_tensor_and_transpose
        mx_tensor, mx_tma_transpose = mx_tensor_and_tranpose
    else:
        x_tensor = x
        w_tensor, w_tma_transpose = w, False
        mx_tensor, mx_tma_transpose = mx_ctx.weight_scale, False

    if isinstance(x_tensor, torch.Tensor):
        x_tensor = flex.lhs_data.reinterpret(x)
    if isinstance(w_tensor, torch.Tensor):
        w_tensor = flex.rhs_data.reinterpret(w)
    kernels = get_kernels(epilogue.specs, fused_activation.specs)

    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(grid,)](
                   flex.out_data.reinterpret(memory["output"]),
                   flex.out_data.reinterpret(out0), *out0.stride(), *out0_flex,
                   x_tensor, x, x.stride(0), x.stride(1), x.stride(2),
                   flex.lhs_data.scale,
                   w_tensor, w.stride(0), w.stride(1), w.stride(2), w_tma_transpose,
                   flex.rhs_data.scale,
                   mx_tensor, mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n, mx_tma_transpose,
                   bias, bias_stride,
                   x.shape[1] if routing_data.expt_hist is None else None,
                   N, K,
                   betas, gammas,
                   None if gather_indx is None else gather_indx.src_indx,
                   None if scatter_indx is None else scatter_indx.src_indx,
                   num_indx,
                   writeback_idxs, writeback_size,
                   expt_hist, expt_token_offs_raw, expt_hist_sum, expt_block_pid_map,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   *fused_activation.fn_args, fused_activation.reduction_n,
                   *epilogue.fn_arg_values_matmul,
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
                   SWIZZLE_MX_VALUE=mx_ctx.swizzle_value.name if mx_ctx.swizzle_value else None,
                   SWIZZLE_MX_SCALE=mx_ctx.swizzle_scale.name if mx_ctx.swizzle_scale else None,
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
                   SWAP_XW=preprocessing_features.swap_xw,
                   NUM_SMS = grid if opt_flags.is_persistent else 0,
                   **opt_flags.target_kernel_kwargs)
    # post-processing
    out = apply_postprocessing_features(scatter_indx, finalize_scatter_idxs, opt_flags, expt_token_offs_raw,
                                num_indx, precision_config, routing_data,
                                postprocessing_features, memory, fused_postprocess_activation, epilogue)
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
