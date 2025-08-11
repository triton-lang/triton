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
from .matmul_ogs_details._matmul_ogs import _compute_writeback_idx
from .matmul_ogs_details._matmul_ogs import _matmul_ogs
from .matmul_ogs_details._p_matmul_ogs import _p_matmul_ogs, get_per_device_per_stream_alloc_fn
from .matmul_ogs_details._finalize_matmul import _finalize_matmul
from .matmul_ogs_details.opt_flags import make_opt_flags, update_opt_flags_constraints
from .numerics_details.mxfp import MXFP_BLOCK_SIZE
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
    DEQUANTIZE_MXFP8 = auto()


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


# ---------------------
# Numerics
# ---------------------

# fmt: off

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
    act_scale: Tensor | None = None
    weight_scale: Tensor| None = None
    out_scale: Tensor | None = None
    out_dtype: torch.dtype = None
    enforce_bitwise_invariance: bool = False

# ---------------------
# Preprocessing
# ---------------------

@dataclass(frozen=True)
class PreprocessingFeatures:
    swap_xw: bool


def init_preprocessing_features(w, precision_config, opt_flags):
    swap_xw = False  # Whether or not to swap X and W operands to the tl.dot
    if target_info.cuda_capability_geq(10, 0):
        swap_xw = precision_config.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
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

        out_scale = precision_config.out_scale
        out_has_mx = out_scale is not None
        out_scale_strides = (None, None) if out_scale is None else out_scale.stride()[-2:]
        mx_a_scale = memory["scratchpad"].get("mx_out_scale", None)
        if mx_a_scale is not None:
            mx_a_scale_stride_k, mx_a_scale_stride_m = [mx_a_scale.stride(i) for i in (0, 2)]
        else:
            mx_a_scale_stride_k, mx_a_scale_stride_m = None, None

        kernels = get_kernels(epilogue.specs, fused_activation.specs)
        kernels._finalize_matmul[grid](
            flex_ctx.out_data.reinterpret(out_scatter),
            *((None, out_scale, None) if out_has_mx else out_scatter_flex),
            *out_scale_strides,
            flex_ctx.out_data.reinterpret(inp), inp.stride(0), inp.stride(2),
            inp_flex.expected_scale if mx_a_scale is None else mx_a_scale,
            mx_a_scale_stride_k, mx_a_scale_stride_m,
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
    M = x.shape[-2]
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
    batch_dim = x.shape[0] if x.ndim == 3 else 1
    y_shape = (batch_dim, y_rows, N // fused_activation.reduction_n)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (y_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    # if we need either standalone scatter or split-k, the matmul output will need post-processing
    if postprocessing_features.finalize:
        if opt_flags.split_k > 1 or not opt_flags.fused_scatter:
            dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
            scratchpad["matmul"] = ((opt_flags.split_k, 1, M, N), dtype)
        if precision_config.out_scale is not None and not (scratchpad.get("matmul", None) is not None and scratchpad["matmul"][1].itemsize > 1):
            scratchpad["mx_out_scale"] = ((opt_flags.split_k, 1, M, triton.cdiv(N, MXFP_BLOCK_SIZE)), torch.uint8)
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
# Canonicalize
# -----------------------------------------------------------------------------
# the `matmul_ogs` kernel can operate on 2D or 3D inputs depending on the mode being used
# we can canonicalize storages to make the implementation more uniform

def _canonicalize_storage(storage, out_ndim, flex_data):
    assert out_ndim >= storage.data.ndim
    new_storage_shape = [1] * (out_ndim - storage.data.ndim) + list(storage.data.shape)
    new_storage_data = storage.data.view(new_storage_shape)
    if flex_data is not None:
        new_storage_data = flex_data.reinterpret(new_storage_data)
    return Storage(new_storage_data, storage.layout)


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
    # determine shapes
    is_ragged = routing_data.expt_hist is not None
    M = x.shape[-2] if gather_indx is None else gather_indx.src_indx.shape[0]
    batch_size = w.shape[0] if routing_data.expt_hist is None and w.ndim == 3 else 1
    K, N = w.shape[-2:]
    assert K == x.shape[-1]
    if x.ndim == 3 and w.ndim == 3:
        assert x.shape[0] == w.shape[0]
    # compute optimization flags
    out_dtype = precision_config.out_dtype or x.dtype
    can_use_tma = x.numel() > 0 and x.storage.is_tma_compliant() and \
                  w.numel() > 0 and w.storage.is_tma_compliant() and \
                 (w_scale is None or w_scale.storage.is_tma_compliant())
    # hopper w/ mxfp4 doesn't support TMA
    can_use_tma = can_use_tma and (torch.cuda.get_device_capability()[0] > 9 or bitwidth(w.dtype) != 4)
    can_use_fused_scatter = scatter_indx is not None and fused_activation.specs.fn is None
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        M, N, K, routing_data, can_use_tma, can_use_fused_scatter, epilogue.effective_itemsize,
    )
    if w_scale is not None and opt_flags.is_persistent and not target_info.has_native_mxfp():
        raise NotImplementedError("Must use non-persistent kernel for simulated MXFP")
    if w_scale is not None and w_scale.storage.layout.name is not None and not opt_flags.is_persistent and target_info.has_native_mxfp():
        raise NotImplementedError("Must use persistent kernel and be TMA-compliant for native MXFP")
    # determine necessary pre/post processing
    preprocessing_features = init_preprocessing_features(w, precision_config, opt_flags)
    postprocessing_features = init_postprocessing_features(routing_data, scatter_indx, opt_flags)
    # allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, fused_activation,
        routing_data, gather_indx, scatter_indx,
        opt_flags, preprocessing_features, postprocessing_features
    )
    memory = apply_allocation(allocation, y)
    if batch_size * M * N == 0:
        ret = memory["output"].squeeze(0)
        if not is_input_batched:
            ret = ret.squeeze(0)
        return ret
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    # Intermediate tensors and postprocess kernels for each situation
    out0, out0_flex = memory["output"], precision_config.flex_ctx.out_data
    fused_postprocess_activation = FusedActivation(FnSpecs.default(), tuple(), 1)
    out_scale = None if precision_config.out_scale is None else precision_config.out_scale.data.view(torch.uint8)
    if postprocessing_features.finalize:
        if opt_flags.fused_scatter:
            out0 = memory["output"]
        else:
            out0 = memory["scratchpad"]["matmul"]
        if "mx_out_scale" in memory["scratchpad"]:
            assert out_scale is not None
            out_scale = memory["scratchpad"]["mx_out_scale"]
        out0_flex = OutFlexData() if out0.dtype == torch.float32 else precision_config.flex_ctx.out_data

        fused_activation, fused_postprocess_activation = fused_postprocess_activation, fused_activation
    out_has_mx = out_scale is not None and out0.element_size() == 1
    if out_has_mx:
        if isinstance(out_scale, Tensor):
            out_scale = Tensor(out_scale)
    else:
        out_scale = None
    # pre-processing
    x, w, writeback_idxs, writeback_size, finalize_scatter_idxs = apply_preprocessing_features(
        x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features
    )
    # matrix multiplication
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]
    # moe metadata
    expt_data = routing_data.expt_data
    block_m = opt_flags.block_m
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map[block_m]
    # spmd grid
    grid_m = triton.cdiv(M, opt_flags.block_m)
    if expt_block_pid_map is not None:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    max_grid = batch_size * grid_m * grid_n * opt_flags.split_k
    grid = min(target_info.num_sms() - opt_flags.idle_sms, max_grid) if opt_flags.is_persistent else max_grid
    # canonicalize storage
    has_gather = gather_indx is not None
    has_scatter = writeback_idxs is not None
    has_gather_tma = has_gather and target_info.has_tma_gather()
    has_scatter_tma = has_scatter and target_info.has_tma_gather()
    y = wrap_torch_tensor(out0.view(math.prod(out0.shape[:-1]), out0.shape[-1]) if has_scatter else out0.view(math.prod(out0.shape[:-2]), *out0.shape[-2:]))
    x_storage = _canonicalize_storage(x.storage, 2 if has_gather_tma else 3, flex.lhs_data)
    w_storage = _canonicalize_storage(w.storage, 3, flex.rhs_data)
    y_storage = _canonicalize_storage(y.storage, 2 if has_scatter_tma else 3, flex.out_data)
    # create tma descriptor for x
    x_has_tma = opt_flags.is_persistent and (has_gather_tma or not has_gather)
    x_tma_block_size = [1, opt_flags.block_k] if has_gather_tma else [1, opt_flags.block_m, opt_flags.block_k]
    x_tma_mode = None if not x_has_tma else "ragged" if is_ragged and not has_gather_tma else "dense"
    x_tensor_or_tma = x_storage.make_tma(x_tma_block_size, x_tma_mode) if x_has_tma else x_storage.data
    # create tma descriptor for y
    y_has_tma = opt_flags.is_persistent and (has_scatter_tma or not has_scatter)
    block_n = opt_flags.block_n // opt_flags.epilogue_subtile // fused_activation.reduction_n
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
    out_scale_strides = out_scale.stride() if out_has_mx else (None, None, None, None)
    out_scale_strides = (0, ) * (3 - len(out_scale_strides)) + out_scale_strides
    # launch kernel
    kernels = get_kernels(epilogue.specs, fused_activation.specs)
    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(grid,)](
                   y_tensor_or_tma, y_storage.data, *out0.stride(),
                   *((None, out_scale, None) if out_has_mx else out0_flex),
                   *out_scale_strides[-3:],
                   x_tensor_or_tma, x_storage.data, *x_strides,
                   flex.lhs_data.scale,
                   None if x_scale is None else x_scale.data.view(torch.uint8), *x_scale_strides,
                   w_tensor_or_tma, w_storage.data, *w_storage.data.stride(), w_storage.data.stride()[-1] != 1,
                   flex.rhs_data.scale,
                   w_scale_tensor_or_tma, *w_scale_strides,
                   bias, bias_stride,
                   x.shape[-2],
                   x.shape[-2] if routing_data.expt_hist is None else None,
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
                   SWIZZLE_MX_VALUE=w.storage.layout.name,
                   SWIZZLE_MX_SCALE=None if w_scale is None else w_scale.storage.layout.name,
                   EPILOGUE_SUBTILE=opt_flags.epilogue_subtile,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=K % opt_flags.block_k == 0,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=routing_data.expected_tokens_per_expt,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(x, w, out0),
                   X_TMA_MODE=x_tma_mode,
                   Y_TMA_MODE=y_tma_mode,
                   SWAP_XW=preprocessing_features.swap_xw,
                   IS_EPILOGUE_DEQUANT_MXFP8=epilogue.specs.name == FnName.DEQUANTIZE_MXFP8.name,
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
    if bias.ndim == 1:
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
