# isort: off
# fmt: off
from dataclasses import dataclass
import itertools
import sys
import torch
import triton
from enum import Enum, auto
import math
# utilities
from triton_kernels import target_info
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.target_info import is_cuda
# details
from .matmul_ogs_details._matmul_ogs import _matmul_ogs
from .matmul_ogs_details._p_matmul_ogs import _p_matmul_ogs, get_per_device_per_stream_alloc_fn
from .matmul_ogs_details._reduce_grouped import _reduce_grouped
from .numerics_details.mxfp import MXFP_BLOCK_SIZE
from .matmul_ogs_details.opt_flags import make_opt_flags, update_opt_flags_constraints, InapplicableConstraint
from .specialize import specialize
from .tensor import Storage, Tensor, FP4, bitwidth, wrap_torch_tensor


@dataclass(frozen=True)
class FnSpecs:
    name: str
    fn: "triton.runtime.jit.JITFunction"
    fn_arg_names: tuple[str]
    fn_arg_do_not_specialize: tuple[str] = tuple()

    @staticmethod
    def default():
        return FnSpecs("dflt", None, tuple())


@dataclass(frozen=True)
class FusedActivation:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object] = tuple()
    reduction_n: int = 1


@dataclass(frozen=True)
class Epilogue:
    specs: FnSpecs = FnSpecs.default()
    fn_arg_values_matmul: tuple[object] = tuple()
    fn_arg_values_finalize: tuple[object] = tuple()
    effective_itemsize: float = None

class FnName(Enum):
    QUANTIZE_MXFP8 = auto()


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
    module._matmul_ogs = specialize(_matmul_ogs, module, spec_constants, spec_tuples,
                                    do_not_specialize=do_not_specialize)
    module._p_matmul_ogs = specialize(_p_matmul_ogs, module, spec_constants, spec_tuples,
                                      do_not_specialize=do_not_specialize)
    module._reduce_grouped = specialize(_reduce_grouped, module, spec_constants, spec_tuples,
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


# This supports computing dw for ragged matmul.  Note the correspondence:
#   fwd pass:      y = matmul_ogs(x, w, ...)
#   bwd pass (dw): dw = matmul_ogs(x.T, dy, ...)
#
# Thus, "our" x, w, and y (as seen by matmul_ogs) correspond to x.T, dy, dw, respectively.
# To avoid confusion, now we'll stick to "x, w, y" terminology.
#
# Assume that y.shape == (N_EXPTS, M, N), x.shape == (M, K), w.shape = (K_W, N).
#
# To make things feasible, we require that x and w satisfy the following condition:
#   (1) We don't support gather/scatter indices: in x, all columns for expt #0 are grouped at
#       the leftmost part, followed by expt #1, and so on.  Ditto for w (top to bottom).
#   (2) At least one of x and w are padded: each expert uses a multiple of block_k columns
#       (or rows), and unused values are filled with zero.
#   (3) No inf or nan are allowed in x or w (except for the final padding - see below).
#       This is because we use "multiplying by padded zero region" in lieu of masking.
#   (4) The number of actually used columns/rows equals self.base.expt_hist.sum() and may be
#       less than K or K_W.  In this case, the final "unused" values can be left uninitialized.
#       However, if x or w is unpadded, the first block_k columns/rows of the unused part must
#       not contain nan or inf.
#
# For example, assume N_EXPTS == 5, block_k == 32, and expt_hist == [60, 33, 0, 32, 25].
#
#               if unpadded     if padded
#               -----------     ---------
#   x: expt #0: x[:, :60]       x[:, :60]
#                               x[:, 60:64] - zero padded
#      expt #1: x[:, 60:93]     x[:, 64:97]
#                               x[:, 97:128] - zero padded
#      expt #3: x[:, 93:125]    x[:, 128:160]
#      expt #4: x[:, 125:150]   x[:, 160:185]
#                               x[:, 185:192] - zero padded
#               x[:, 150:min(182, K)] - must not contain inf/nan
#
#               x[:, 182:]      x[:, 192:] - unused (may contain garbage, including inf/nan)
#
#   w is the same, except that rows columns are flipped.
@dataclass
class InnerRoutingData:
    base: RoutingData | None = None
    block_k: int | None = None
    x_is_padded: bool = False
    w_is_padded: bool = False

    # Return value contains: ExptHist, ExptOffs, ExptTileOffs, ExptData,
    #                        EXPT_IS_INNER, X_IS_PADDED, W_IS_PADDED, ExptHistMax
    @staticmethod
    def make_kernel_args(data, block_m):
        if isinstance(data, RoutingData):
            expt_data, block = data.expt_data, block_m
            args = (False, False, False, None)
        elif isinstance(data, InnerRoutingData):
            expt_data, block = data.base.expt_data, data.block_k
            args = (
                True, data.x_is_padded, data.w_is_padded, expt_data.hist.max()
            )
        elif data is None:
            expt_data = None
        else:
            assert None

        if expt_data is None:
            return (None, None, None, None, False, False, False, None)

        return (
            expt_data.hist,
            expt_data.token_offs_raw,
            expt_data.token_offs_pad(block),
            expt_data.block_pid_map(block),
        ) + args


# ---------------------
# Numerics
# ---------------------

# fmt: off

@dataclass(frozen=True)
class FlexCtx:
    lhs_data: InFlexData = InFlexData()
    rhs_data: InFlexData = InFlexData()
    out_data: OutFlexData = OutFlexData()
    acc_data: InFlexData = InFlexData()

@dataclass
class PrecisionConfig:
    max_num_imprecise_acc: int = None
    allow_tf32: bool = True
    flex_ctx: FlexCtx = FlexCtx()
    acc_scale: int = 1.0
    flexpoint_saturate_inf: bool = False
    report_quantization_err_fn: callable = None
    act_scale: Tensor | None = None
    weight_scale: Tensor| None = None
    out_scale: Tensor | None = None
    out_dtype: torch.dtype = None
    enforce_bitwise_invariance: bool = False


# TODO: merge in opt_flags
def get_swap_xw(precision_config, opt_flags):
    if target_info.cuda_capability_geq(10, 0):
        return precision_config.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
    return False

# ---------------------
# Allocation
# ---------------------

@dataclass
class MatmulAllocation:
    device: str
    output: tuple[tuple[int], torch.dtype]
    scratchpads: dict[str, tuple]

def init_allocation(x, w, precision_config, fused_activation,
                    routing_data, gather_indx, scatter_indx, inner_routing_data, opt_flags):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.src_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        Mc = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
        y_rows = Mc
    if inner_routing_data is not None:
        batch_dim = inner_routing_data.base.n_expts_tot
    else:
        batch_dim = x.shape[0] if x.ndim == 3 else 1
    out_shape = (batch_dim, y_rows, N // fused_activation.reduction_n)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (out_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    if opt_flags.split_k > 1 or (scatter_indx is not None and not opt_flags.fused_scatter):
        scratch_out_dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
        scratchpad["matmul"] = ((opt_flags.split_k, 1, M, N), scratch_out_dtype)
    if "matmul" in scratchpad and precision_config.out_scale is not None:
        scratchpad["mx_out_scale"] = ((opt_flags.split_k, 1, M, triton.cdiv(N, MXFP_BLOCK_SIZE)), torch.uint8)
    return MatmulAllocation(x.device, output, scratchpad)

def apply_allocation(allocation: MatmulAllocation, output):
    ret = dict()
    if output is None:
        output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
    else:
        if output.ndim == 2:
            output = output[None, :, :]
        assert output.shape == allocation.output[0]
    ret["output"] = output[None, :, :]
    ret["scratchpad"] = {
        k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            for k, v in allocation.scratchpads.items()
    }
    return ret

# -----------------------------------------------------------------------------
# Canonicalize
# -----------------------------------------------------------------------------
# the `matmul_ogs` kernel can operate on 2D or 3D inputs depending on the mode being used
# we can canonicalize storages to make the implementation more uniform

def _canonicalize_storage(storage, out_ndim, flex_data):
    assert out_ndim >= storage.data.ndim
    # Need to use as_strided instead of view because for a tensor with
    # shape[-2] == 1 can have ambuiguity related to col-wise. Fo example,
    # > t = torch.randn(2, 5, 1).mT
    # > t_view = t.view(t.shape)
    # > t.stride(), t_view.stride()
    # ((5, 1, 1), (5, 5, 1))
    # Our check t_view is col-wise fails since t_view.stride(-2) != 1
    # This case is covered by (m, n, k) == (1000, 700, 2) in test_matmul.py
    new_storage_shape = [1] * (out_ndim - storage.data.ndim) + list(storage.data.shape)
    new_storage_stride = [0] * (out_ndim - storage.data.ndim) + list(storage.data.stride())
    new_storage_data = storage.data.as_strided(new_storage_shape, new_storage_stride)
    if flex_data is not None:
        new_storage_data = flex_data.reinterpret(new_storage_data)
    return Storage(new_storage_data, storage.layout)

#

def reduce_grouped(x: torch.Tensor, indx: torch.Tensor, out: torch.Tensor, out_mx_scale: torch.Tensor,
                   fused_activation, epilogue,
                   x_flex: InFlexData | None = None,
                   out_flex: OutFlexData | None = None, x_mx_scale: torch.Tensor | None = None,
                   out_dtype: bool = None, flexpoint_saturate_inf: bool = False):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]` and overwrites the row corresponding to the first
    valid (non-negative) index with the per-group sum. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0), None
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    if x_flex is None:
        x_flex = InFlexData()
    if out_flex is None:
        out_flex = OutFlexData()
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % fused_activation.reduction_n == 0
    BLOCK_N = 512
    # Resolve scalar flex scales (may be None)
    x_expected_scale = None if x_flex is None else x_flex.scale
    out_expected_scale = None if out_flex is None else out_flex.expected_scale
    out_actual_scale = None if out_flex is None else out_flex.actual_scale
    out_checksum_scale = None if out_flex is None else out_flex.checksum_scale
    # Resolve MXFP output scale row stride
    stride_mxb = 0 if x_mx_scale is None else x_mx_scale.stride(0)
    stride_mxs = 0 if x_mx_scale is None else x_mx_scale.stride(1)
    stride_omxs = 0 if out_mx_scale is None else out_mx_scale.stride(0)
    kernels = get_kernels(epilogue.specs, fused_activation.specs)
    kernels._reduce_grouped[(num_groups, )](
        x_flex.reinterpret(x), x.stride(0), x.stride(2), x.stride(3),  #
        x_expected_scale,  # scalar input scale
        out_flex.reinterpret(out), out.stride(1), out.stride(2),  #
        out_expected_scale, out_actual_scale, out_checksum_scale, indx,  #
        x.shape[0], x.shape[-1],  #
        x_mx_scale, stride_mxb, stride_mxs,  #
        out_mx_scale, stride_omxs,  #
        *fused_activation.fn_args, fused_activation.reduction_n,
        *epilogue.fn_arg_values_finalize,
        HAS_IN_MX_SCALE=x_mx_scale is not None, HAS_OUT_MX_SCALE=out_mx_scale is not None,
        FLEXPOINT_SATURATE_INF=flexpoint_saturate_inf,  #
        BLOCK_N=BLOCK_N, K=K,  #
        num_warps=1,  #
    )
    return out, out_mx_scale

# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------

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
               y_acc_in: torch.Tensor | None = None,
               inner_routing_data: InnerRoutingData | None = None,
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
        assert inner_routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if inner_routing_data is not None:
        assert routing_data is None
        assert gather_indx is None
        assert scatter_indx is None
        routing_data = RoutingData(None, None, inner_routing_data.base.n_expts_tot, 1)
    # canonicalize inputs
    if precision_config is None:
        precision_config = PrecisionConfig()
    if fused_activation is None:
        fused_activation = FusedActivation(FnSpecs.default(), tuple(), 1)
    if epilogue is None:
        epilogue = Epilogue(FnSpecs.default(), tuple(), tuple(), False)
    if routing_data is None:
        routing_data = RoutingData(None, None, max(1, w.shape[0]), 1)
    # unpack scales
    w_scale = precision_config.weight_scale
    w_has_mx = w_scale is not None
    is_hopper_fp8 = is_cuda() and not target_info.cuda_capability_geq(10, 0) and bitwidth(w.dtype) == 8
    if w_has_mx: assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    if is_hopper_fp8: assert w.stride(-2) == 1, "`w` must be column-major when it has data-type FP8 on capability < 10"
    if not isinstance(w, Tensor):
        # TODO: remove this code path; using uint8 for mxfp4 weight will bite us when we want to support uint8 for real
        dtype = FP4 if w.dtype == torch.uint8 else w.dtype
        w = wrap_torch_tensor(w, dtype=dtype)
    if w_scale is not None and not isinstance(w_scale, Tensor):
        w_scale = Tensor(w_scale)
    if w_scale is not None:
        w_scale.storage.data = w_scale.data.view(torch.uint8)
        w_scale.dtype = torch.uint8
    x_scale = precision_config.act_scale
    x_has_mx = x_scale is not None
    if x_has_mx: assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_scale is not None and not isinstance(x_scale, Tensor):
        x_scale = Tensor(x_scale)
    if not isinstance(x, Tensor):
        x = Tensor(x, dtype=x.dtype)
    x_transpose = x.stride(-1) != 1
    # determine shapes
    has_gather = gather_indx is not None
    has_scatter = scatter_indx is not None
    is_ragged = routing_data.expt_hist is not None
    M = x.shape[-2] if gather_indx is None else gather_indx.src_indx.shape[0]
    if inner_routing_data is not None:
        batch_size = inner_routing_data.base.n_expts_tot
    else:
        batch_size = w.shape[0] if routing_data.expt_hist is None and w.ndim == 3 else 1
    if y_acc_in is not None:
        y_acc_is_y = y_acc_in.data_ptr() == y.data_ptr() and y_acc_in.stride() == y.stride()
    else:
        y_acc_is_y = None
    K = x.shape[-1]
    K_W, N = w.shape[-2:]
    if x.ndim == 3 and w.ndim == 3:
        assert x.shape[0] == w.shape[0]
    # compute optimization flags
    out_dtype = precision_config.out_dtype or x.dtype
    can_use_tma = (
        x.numel() > 0 and x.storage.is_tma_compliant() and
        w.numel() > 0 and w.storage.is_tma_compliant() and
        (w_scale is None or w_scale.storage.is_tma_compliant()) and
        (not is_ragged or x.stride(-1) == 1) and
        # Currently we don't support tma if y is column major; may revisit later if this becomes an issue.
        (y is None or y.stride(-1) == 1) and
        (y_acc_in is None or y_acc_is_y) and
        # If we use inner_routing_data, w must be either padded or row major, otherwise we get
        # unaligned access.
        (inner_routing_data is None or w.stride(-1) == 1 or inner_routing_data.w_is_padded)
    )
    has_gather_tma = has_gather and target_info.has_tma_gather()
    # hopper w/ mxfp4 doesn't support TMA
    can_use_tma = can_use_tma and (torch.cuda.get_device_capability()[0] > 9 or bitwidth(w.dtype) != 4)
    can_use_fused_scatter = has_scatter and (fused_activation.specs.fn is None) and (epilogue.specs.fn is None) and (routing_data.n_expts_act == 1)
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        batch_size, M, N, w.shape[-2], routing_data,
        can_use_tma, can_use_fused_scatter, epilogue.effective_itemsize,
        x_transpose, y_acc_in is not None,
        inner_routing_data.block_k if inner_routing_data is not None else None,
    )
    if not can_use_fused_scatter and opt_flags.fused_scatter:
        raise InapplicableConstraint("Fused scatter is not supported")
    if inner_routing_data is not None:
        assert opt_flags.block_k == inner_routing_data.block_k
        assert opt_flags.split_k == 1
        batch_size = inner_routing_data.base.n_expts_tot
        # For unpadded (row major) x, we cannot use tma because memory access isn't aligned.
        x_has_tma = opt_flags.is_persistent and (x.stride(-1) != 1 or inner_routing_data.x_is_padded)
        # If TMA is used, limit is handled automatically, so we can pretend K is "even".
        # (For unpadded input, we assume that the first block_k unused rows are zero-filled,
        # when routing_data.expt_hist.sum() is less than K or K_W.)
        if opt_flags.is_persistent:
            even_K = x_has_tma or inner_routing_data.x_is_padded
        else:
            even_K = inner_routing_data.x_is_padded and inner_routing_data.w_is_padded
    else:
        batch_size = w.shape[0] if routing_data.expt_hist is None and w.ndim == 3 else 1
        assert K == K_W
        x_has_tma = opt_flags.is_persistent and (has_gather_tma or not has_gather)
        even_K = (K % opt_flags.block_k == 0)
    if w_scale is not None and opt_flags.is_persistent and not target_info.has_native_mxfp():
        raise NotImplementedError("Must use non-persistent kernel for simulated MXFP")
    if w_scale is not None and w_scale.storage.layout.name is not None and not opt_flags.is_persistent and target_info.has_native_mxfp():
        raise NotImplementedError("Must use persistent kernel and be TMA-compliant for native MXFP")
    # fused activation
    matmul_fused_activation = fused_activation
    reduce_fused_activation = FusedActivation()
    if opt_flags.split_k > 1  or (scatter_indx is not None and not opt_flags.fused_scatter):
        matmul_fused_activation, reduce_fused_activation = reduce_fused_activation, matmul_fused_activation
    # allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, fused_activation,
        routing_data, gather_indx, scatter_indx, inner_routing_data, opt_flags)
    memory = apply_allocation(allocation, y)
    # early exit
    if batch_size * M * N == 0:
        ret = memory["output"].squeeze(0)
        if not is_input_batched:
            ret = ret.squeeze(0)
        return ret
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    # Intermediate tensors and postprocess kernels for each situation
    has_scratchpad = "matmul" in memory["scratchpad"]
    # Canonical output tensor (matmul scratchpad if present, otherwise final output tensor)
    out_matmul = memory["scratchpad"].get("matmul", memory["output"])
    out_matmul_flex = OutFlexData() if out_matmul.dtype == torch.float32 else precision_config.flex_ctx.out_data
    # Unified mx-scale pointer; when scratchpad exists, prefer its mx buffer
    out_matmul_scale = precision_config.out_scale
    if out_matmul_scale is not None:
        out_matmul_scale = out_matmul_scale.data.view(torch.uint8)
        if has_scratchpad and "mx_out_scale" in memory["scratchpad"]:
            out_matmul_scale = memory["scratchpad"]["mx_out_scale"]
    out_matmul_has_mx = out_matmul_scale is not None and out_matmul.element_size() == 1
    # matrix multiplication
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]
    # moe metadata
    block_m = opt_flags.block_m
    expt_data_args = InnerRoutingData.make_kernel_args(inner_routing_data or routing_data, block_m)
    # spmd grid
    grid_m = triton.cdiv(M, opt_flags.block_m)
    if routing_data.expt_data is not None:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    max_grid = batch_size * grid_m * grid_n * opt_flags.split_k
    grid = min(target_info.num_sms() - opt_flags.idle_sms, max_grid) if opt_flags.is_persistent else max_grid
    # canonicalize storage
    has_scatter_tma = opt_flags.fused_scatter and target_info.has_tma_gather()
    y = wrap_torch_tensor(out_matmul.view(math.prod(out_matmul.shape[:-1]), out_matmul.shape[-1]) if opt_flags.fused_scatter else out_matmul.view(math.prod(out_matmul.shape[:-2]), *out_matmul.shape[-2:]))
    x_storage = _canonicalize_storage(x.storage, 2 if has_gather_tma else 3, flex.lhs_data)
    w_storage = _canonicalize_storage(w.storage, 3, flex.rhs_data)
    y_storage = _canonicalize_storage(y.storage, 2 if has_scatter_tma else 3, flex.out_data)
    # create tma descriptor for x
    if y_acc_in is not None:
        assert opt_flags.split_k == 1, "y_acc_in + split_k is not supported."
        assert scatter_indx is None, "y_acc_in + scatter is not supported."
        if y_acc_in.ndim == 2:
            y_acc_in = y_acc_in.unsqueeze(0)
        assert y_acc_in.shape == out_matmul.shape[-3:]
        y_acc_strides = y_acc_in.stride()
    else:
        y_acc_strides = (None, None, None)

    x_tma_block_size = [1, opt_flags.block_k] if has_gather_tma else [1, opt_flags.block_m, opt_flags.block_k]
    x_tma_mode = None if not x_has_tma else "ragged" if is_ragged and not has_gather_tma else "dense"
    x_tensor_or_tma = x_storage.make_tma(x_tma_block_size, x_tma_mode) if x_has_tma else x_storage.data
    # create tma descriptor for y
    y_has_tma = (
        opt_flags.is_persistent and (has_scatter_tma or not opt_flags.fused_scatter)
        and (y_acc_in is None or y_acc_is_y)
    )
    block_n = opt_flags.block_n // opt_flags.epilogue_subtile // matmul_fused_activation.reduction_n
    y_tma_block_size = [1, block_n] if has_scatter_tma else [1, opt_flags.block_m, block_n]
    y_tma_mode = None if not y_has_tma else "ragged" if is_ragged and not has_scatter_tma else "dense"
    y_tensor_or_tma = y_storage.make_tma(y_tma_block_size, y_tma_mode) if y_has_tma else y_storage.data
    # create tma descriptor for w
    w_has_tma = opt_flags.is_persistent
    w_tensor_or_tma = w_storage.make_tma([1, opt_flags.block_k, opt_flags.block_n], "dense") if w_has_tma else w_storage.data
    # create tma descriptor for w_scale
    w_scale_tensor_or_tma = w_scale
    w_scale_has_tma = opt_flags.is_persistent and w_scale is not None
    w_scale_tensor_or_tma =  w_scale.storage.make_tma([opt_flags.block_n, opt_flags.block_k], "dense") if w_scale_has_tma else w_scale
    # canonicalize strides
    x_strides = [0]*(3 - x_storage.data.ndim) + list(x_storage.data.stride())
    x_scale_strides = x_scale.stride() if x_has_mx else (None, None, None)
    x_scale_strides = (0, ) * (3 - len(x_scale_strides)) + x_scale_strides
    w_scale_strides = w_scale.stride() if w_has_mx and not w_scale_has_tma else (None, None, None)
    w_scale_strides = (0, ) * (3 - len(w_scale_strides)) + w_scale_strides
    out_matmul_scale_strides = out_matmul_scale.stride() if out_matmul_has_mx else (None, None, None, None)
    out_matmul_scale_strides = (0, ) * (4 - len(out_matmul_scale_strides)) + out_matmul_scale_strides
    # launch kernel
    kernels = get_kernels(epilogue.specs, matmul_fused_activation.specs)
    # When stride(-2) == stride(-1) == 1, it's ambiguous whether W is transposed
    # (i.e. col-wise). Since this matters when w_has_mx is True and w_transpose
    # is True the fast code path, stride(-2) == 1 takes precedence, e.g., vs.
    # w_transpose = w_storage.data.stride()[-1] != 1
    w_transpose = w_storage.data.stride()[-2] == 1

    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(grid,)](
                   y_tensor_or_tma, y_storage.data, *out_matmul.stride(),
                   *((None, out_matmul_scale, None) if out_matmul_has_mx else out_matmul_flex),
                   *out_matmul_scale_strides[-4:],
                   x_tensor_or_tma, x_storage.data, *x_strides, x_transpose,
                   flex.lhs_data.scale,
                   None if x_scale is None else x_scale.data.view(torch.uint8), *x_scale_strides,
                   w_tensor_or_tma, w_storage.data, *w_storage.data.stride(), w_transpose,
                   flex.rhs_data.scale,
                   w_scale_tensor_or_tma, *w_scale_strides,
                   flex.acc_data.reinterpret(y_acc_in), *y_acc_strides,
                   flex.acc_data.scale, y_acc_is_y,
                   bias, bias_stride,
                   x.shape[-2] if routing_data.expt_hist is None else None,
                   N, K, K_W,
                   betas, gammas,
                   None if gather_indx is None else gather_indx.src_indx,
                   None if scatter_indx is None else scatter_indx.src_indx,
                   num_indx,
                   None if not opt_flags.fused_scatter else scatter_indx.dst_indx,
                   None if not opt_flags.fused_scatter else scatter_indx.dst_indx.shape[0],
                   *expt_data_args,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   *matmul_fused_activation.fn_args, matmul_fused_activation.reduction_n,
                   *epilogue.fn_arg_values_matmul,
                   routing_data.n_expts_tot, routing_data.n_expts_act,
                   precision_config.max_num_imprecise_acc,
                   precision_config.allow_tf32,
                   precision_config.flexpoint_saturate_inf,
                   flex.rhs_data.is_per_batch,
                   flex.out_data.is_per_batch,
                   flex.acc_data.is_per_batch,
                   opt_flags.block_m,
                   opt_flags.block_n,
                   opt_flags.block_k,
                   opt_flags.group_m,
                   XCD_SWIZZLE=opt_flags.xcd_swizzle,
                   SWIZZLE_MX_VALUE=w.storage.layout.name,
                   SWIZZLE_MX_SCALE=None if w_scale is None else w_scale.storage.layout.name,
                   EPILOGUE_SUBTILE=opt_flags.epilogue_subtile,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=even_K,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=routing_data.expected_tokens_per_expt,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(x, w, out_matmul),
                   X_TMA_MODE=x_tma_mode,
                   Y_TMA_MODE=y_tma_mode,
                   SWAP_XW=get_swap_xw(precision_config, opt_flags),
                   IS_EPILOGUE_QUANT_MXFP8=epilogue.specs.name == FnName.QUANTIZE_MXFP8.name,
                   NUM_SMS = grid if opt_flags.is_persistent else 0,
                   **opt_flags.target_kernel_kwargs)
    # Build grouped reduction inputs in a uniform way
    group_indx = None if scatter_indx is None or opt_flags.fused_scatter else scatter_indx.src_indx.view(-1, routing_data.n_expts_act)
    out_final, out_final_mx_scale = reduce_grouped(
        out_matmul,
        group_indx,
        memory["output"].squeeze(0),
        precision_config.out_scale,
        reduce_fused_activation,
        epilogue,
        x_flex=InFlexData(dtype=out_matmul_flex.dtype, scale=out_matmul_flex.expected_scale),
        out_flex=precision_config.flex_ctx.out_data,
        x_mx_scale=out_matmul_scale.squeeze(1) if out_matmul_has_mx else None,
        out_dtype=memory["output"].dtype,
        flexpoint_saturate_inf=precision_config.flexpoint_saturate_inf,
    )
    if not (is_input_batched or inner_routing_data is not None):
        out_final = out_final.squeeze(0)
    if out_final_mx_scale is not None:
        precision_config.out_scale = out_final_mx_scale
    return out_final

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
                 inner_routing_data: InnerRoutingData | None = None,
                 round_x = None, round_y = None,
                 ):
    if inner_routing_data is not None:
        assert bias is None, "Not supported yet"
        m, n = x.shape[-2], w.shape[-1]
        block_k = inner_routing_data.block_k
        n_expts_tot = inner_routing_data.base.n_expts_tot
        out = torch.zeros((n_expts_tot, m, n), dtype=torch.float32, device=x.device)
        start_x = start_w = 0
        for expt in range(n_expts_tot):
            k = inner_routing_data.base.expt_hist[expt].item()
            if k > 0:
                out[expt] = matmul_ogs_torch(
                    x[:, start_x:start_x+k], w[start_w:start_w+k, :], None,
                    None, None, None, None, betas, gammas, None, round_x, round_y
                )
            padded_k = triton.cdiv(k, block_k) * block_k
            start_x += padded_k if inner_routing_data.x_is_padded else k
            start_w += padded_k if inner_routing_data.w_is_padded else k
        return out

    is_input_batched = x.ndim == 3
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if round_x is None:
        round_x = lambda x, idx: x
    if round_y is None:
        round_y = lambda x: x
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    if x.ndim == 2:
        x = x.view(1, *x.shape)
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
                           w[i].float())
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
