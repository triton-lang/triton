# isort: off
# fmt: off
from dataclasses import dataclass
import itertools
import torch
import triton
from enum import Enum, auto
import math
from typing import Callable
# utilities
from triton_kernels import target_info
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.target_info import is_cuda
from triton_kernels.tensor_details.layout_details.hopper_scale import HopperMXScaleLayout
# details
from .matmul_details._matmul import _matmul
from .matmul_details._p_matmul import _p_matmul, get_per_device_per_stream_alloc_fn
from .numerics_details.mxfp import MXFP_BLOCK_SIZE
from .tensor_details.layout_details.strided import StridedLayout
from .tensor_details.layout_details.blackwell_scale import BlackwellActMXScaleLayout
from .matmul_details.opt_flags import make_opt_flags, update_opt_flags_constraints
from .specialize import FnSpecs, SpecializationModule, ClosureArg
from .tensor import Storage, Tensor, FP4, bitwidth, wrap_torch_tensor, RaggedTensorMetadata
from .reduce import reduce
from .reduce import PostprocessFn as ReducePostprocessFn
from .tensor_details.ragged_tensor import ragged_metadata_fields

@dataclass(frozen=True)
class FusedActivation:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object, ...] = tuple()


@dataclass(frozen=True)
class Epilogue:
    specs: FnSpecs = FnSpecs.default()
    fn_arg_values_matmul: tuple[object, ...] = tuple()
    fn_arg_values_finalize: tuple[object, ...] = tuple()
    effective_itemsize: float | None = None

class FnName(Enum):
    QUANTIZE_MXFP8 = auto()


@dataclass(frozen=True)
class FusedComm:
    out_handles: torch.Tensor
    scatter_shard_indx: torch.Tensor | None = None
    reduce_rank: int = 0
    n_reduce_shards: int = 1

specializations = SpecializationModule("matmul",
    kernels=[("_matmul", _matmul), ("_p_matmul", _p_matmul)],
    closure_args={
        "epilogue": ClosureArg("EPILOGUE_FN", "epilogue_fn_args"), #
        "activation": ClosureArg("ACTIVATION_FN", "activation_fn_args"), #
    },
)
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
    acc_data: InFlexData = InFlexData()

@dataclass
class PrecisionConfig:
    max_num_imprecise_acc: int | None = None
    allow_tf32: bool = True
    flex_ctx: FlexCtx = FlexCtx()
    acc_scale: float = 1.0
    flexpoint_saturate_inf: bool = False
    report_quantization_err_fn: Callable | None = None
    a_mx_scale: torch.Tensor | Tensor | None = None
    b_mx_scale: torch.Tensor | Tensor | None = None
    c_mx_scale: torch.Tensor | Tensor | None = None
    out_dtype: torch.dtype | None = None
    enforce_bitwise_invariance: bool = False


# TODO: merge in opt_flags
def get_swap_xw(precision_config, opt_flags):
    if target_info.cuda_capability_geq(10, 0):
        return precision_config.b_mx_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
    elif target_info.cuda_capability_geq(9, 0):
        return precision_config.b_mx_scale is not None and opt_flags.is_persistent

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
                    gather_indx, scatter_indx, batch_dim,
                    n_reduce_shards, opt_flags):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    if scatter_indx is not None:
        M = scatter_indx.shape[0]
    y_rows = M
    y_rows *= n_reduce_shards
    out_shape = (batch_dim, y_rows, N // fused_activation.specs.reduction_n)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (out_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    N_scratch = N // fused_activation.specs.reduction_n if opt_flags.split_k == 1 else N
    if opt_flags.split_k > 1:
        scratch_out_dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
        scratchpad["matmul"] = ((opt_flags.split_k, batch_dim, M, N_scratch), scratch_out_dtype)
    if "matmul" in scratchpad and precision_config.c_mx_scale is not None:
        assert batch_dim == 1, "batch_dim > 1 not supported yet"
        scratchpad["mx_c_mx_scale"] = ((opt_flags.split_k, 1, M, triton.cdiv(N_scratch, MXFP_BLOCK_SIZE)), torch.uint8)
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
# the `matmul` kernel can operate on 2D or 3D inputs depending on the mode being used
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


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------

def matmul_set_idle_sms(num_idle_sms):
    """
    persistent kernels will leave `num_idle_sms` idle
    """
    update_opt_flags_constraints({"idle_sms": num_idle_sms})

def matmul(a, b, bias,
    a_ragged_metadata: RaggedTensorMetadata | None = None,
    b_ragged_metadata: RaggedTensorMetadata | None = None,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    precision_config: PrecisionConfig | None = None,
    betas: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    out_alpha: float | None = None,
    c: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
    fused_activation: FusedActivation | None = None,
    epilogue: Epilogue | None = None,
    c_acc_in: torch.Tensor | None = None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

    matmul can be optionally fused with all gather or scatter at the end for the output. When fused_comm is specified, the m-th row of the output will be stored to (m * n_reduce_shards + reduce_rank) -th row
    of each rank id in range [scatter_shard_indx[m] * n_reduce_shards, (scatter_shard_indx[m] + 1) * n_reduce_shards) if scatter_shard_indx is not None, otherwise the output will be all gathered across all reduce ranks.
    When scatter_shard_indx is specified, the caller should ensure that the indices of different shards do not conflict.

    The output buffer for fused comm should be pre-allocated and passed in via fused_comm.out_handles, which contains ipc handles to the output tensors, each with shape (n_rows * n_reduce_shards, n_cols).
    """
    is_input_batched = a.ndim == 3
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert b_ragged_metadata is None, "w cannot be ragged in batched mode"
        assert a_ragged_metadata is None, "x cannot be ragged in batched mode"
        assert fused_comm is None, "fused comm is not supported in batched mode"
        assert b.ndim == 3 and b.shape[0] == a.shape[0]
    if b_ragged_metadata is not None:
        assert gather_indx is None
        assert scatter_indx is None
    # canonicalize inputs
    if precision_config is None:
        precision_config = PrecisionConfig()
    if fused_activation is None:
        fused_activation = FusedActivation(FnSpecs.default(), tuple())
    if epilogue is None:
        epilogue = Epilogue(FnSpecs.default(), tuple(), tuple(), False)
    n_slices = max(1, b.shape[0]) if a_ragged_metadata is None else a_ragged_metadata.n_slices
    # unpack b scale
    b_scale = precision_config.b_mx_scale
    b_has_mx = b_scale is not None
    is_hopper_fp8 = is_cuda() and not target_info.cuda_capability_geq(10, 0) and bitwidth(b.dtype) == 8
    if is_hopper_fp8: assert b.stride(-2) == 1, "`w` must be column-major when it has data-type FP8 on capability < 10"
    if not isinstance(b, Tensor):
        # TODO: remove this code path; using uint8 for mxfp4 weight will bite us when we want to support uint8 for real
        dtype = FP4 if b.dtype == torch.uint8 else b.dtype
        b = wrap_torch_tensor(b, dtype=dtype)
    if b_has_mx and (torch.cuda.get_device_capability()[0] < 10 or b.storage.layout is not None and not isinstance(b.storage.layout, StridedLayout)):
        assert b.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp and (swizzled or not on >=Blackwell)"
    if b_scale is not None and not isinstance(b_scale, Tensor):
        b_scale = Tensor(b_scale)
    if b_scale is not None:
        b_scale.storage.data = b_scale.data.view(torch.uint8)
        b_scale.dtype = torch.uint8
    # unpack a scale
    a_scale = precision_config.a_mx_scale
    a_has_mx = a_scale is not None
    if a_has_mx: assert a.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if a_scale is not None and not isinstance(a_scale, Tensor):
        a_scale = Tensor(a_scale)
    if not isinstance(a, Tensor):
        a = Tensor(a, dtype=a.dtype)
    a_transpose = a.stride(-1) != 1
    # determine shapes
    has_gather = gather_indx is not None
    has_scatter = scatter_indx is not None
    is_a_ragged = a_ragged_metadata is not None
    is_b_ragged = b_ragged_metadata is not None
    is_c_ragged = is_a_ragged and b_ragged_metadata is None
    ragged_dimension = "K" if is_b_ragged else "M" if is_a_ragged else None
    M = a.shape[-2] if gather_indx is None else gather_indx.shape[0]
    if ragged_dimension == "K":
        batch_size = b_ragged_metadata.n_slices
    elif ragged_dimension is None and b.ndim == 3:
        batch_size = b.shape[0]
    else:
        batch_size = 1
    if c_acc_in is not None:
        c_acc_is_c = c_acc_in.data_ptr() == c.data_ptr() and c_acc_in.stride() == c.stride()
    else:
        c_acc_is_c = None
    K = a.shape[-1]
    K_W, N = b.shape[-2:]
    if a.ndim == 3 and b.ndim == 3:
        assert a.shape[0] == b.shape[0]
    # compute optimization flags
    out_dtype = precision_config.out_dtype or a.dtype
    can_use_tma = (
        a.numel() > 0 and a.storage.is_tma_compliant() and
        b.numel() > 0 and b.storage.is_tma_compliant() and
        (b_scale is None or b_scale.storage.is_tma_compliant()) and
        (ragged_dimension != "M" or a.stride(-1) == 1) and
        # Currently we don't support tma if y is column major; may revisit later if this becomes an issue.
        (c is None or c.stride(-1) == 1) and
        (c_acc_in is None or c_acc_is_c) and
        # if ragged dimension is K, w must be either padded or row major to ensure alignment
        (ragged_dimension != "K" or b.stride(-1) == 1 or b_ragged_metadata.slice_sizes_divisibility is not None)
    )
    if b_scale is not None and isinstance(b_scale.storage.layout, StridedLayout) and b_scale.storage.data.stride()[-1] != 1:
        # In this case, we need to transpose b_scale. Then the reduction dim
        # becomes the last dim that will be divided by 32. This to be a multiple
        # of 16 to be TMA-compliant requires block_k to be a multiple of 512,
        # which is too big.
        can_use_tma = False
    has_gather_tma = has_gather and target_info.has_tma_gather()
    can_use_split_k = scatter_indx is None and not a_has_mx and not b_has_mx and ragged_dimension != "K"
    block_k = None
    if ragged_dimension == "K":
        block_k = a_ragged_metadata.slice_sizes_divisibility or b_ragged_metadata.slice_sizes_divisibility
    opt_flags = make_opt_flags(out_dtype, a.dtype, b.dtype, precision_config,
        batch_size, M, N, b.shape[-2], a_ragged_metadata,
        can_use_tma, can_use_split_k, epilogue.effective_itemsize,
        a_transpose, c_acc_in is not None,
        block_k = block_k,
    )
    # there seems to be a bug on A100
    # pytest -vs test_matmul.py::test_op[False-False-False-False-pad_b-16-768-512-1024-ragged-float16-float16-10-1-False-None-False-False-False-True-None]
    if ragged_dimension == "K" and torch.cuda.get_device_capability()[0] < 9:
        opt_flags.num_stages = 1
    if ragged_dimension == "K":
        a_has_tma = opt_flags.is_persistent and (a.stride(-1) != 1 or (a_ragged_metadata.slice_sizes_divisibility is not None))
        # If TMA is used, limit is handled automatically, so we can pretend K is "even".
        # (For unpadded input, we assume that the first block_k unused rows are zero-filled,
        # when routing_data.expt_hist.sum() is less than K or K_W.)
        if opt_flags.is_persistent:
            even_K = a_has_tma or (a_ragged_metadata.slice_sizes_divisibility is not None)
        else:
            even_K = a_ragged_metadata.slice_sizes_divisibility is not None and b_ragged_metadata.slice_sizes_divisibility is not None
    else:
        batch_size = b.shape[0] if a_ragged_metadata is None and b.ndim == 3 else 1
        assert K == K_W
        a_has_tma = opt_flags.is_persistent and (has_gather_tma or not has_gather)
        even_K = (K % opt_flags.block_k == 0)
    if b_scale is not None and b_scale.storage.layout.name is not None and not opt_flags.is_persistent and target_info.has_native_mxfp():
        raise NotImplementedError("Must use persistent kernel and be TMA-compliant for native MXFP")
    # fused activation
    matmul_fused_activation = fused_activation
    reduce_fused_activation = FusedActivation()
    if opt_flags.split_k > 1:
        matmul_fused_activation, reduce_fused_activation = reduce_fused_activation, matmul_fused_activation
    # allocate output/scratchpad memory
    allocation = init_allocation(a, b, precision_config, fused_activation,
                                 gather_indx, scatter_indx, batch_size,
                                 fused_comm.n_reduce_shards if fused_comm is not None else 1,
                                 opt_flags)
    memory = apply_allocation(allocation, c)
    # early exit
    if batch_size * M * N == 0:
        ret = memory["output"].squeeze(0)
        if not is_input_batched:
            ret = ret.squeeze(0)
        return ret
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(a.device))
    # Intermediate tensors and postprocess kernels for each situation
    has_scratchpad = "matmul" in memory["scratchpad"]
    # Canonical output tensor (matmul scratchpad if present, otherwise final output tensor)
    out_matmul = memory["scratchpad"].get("matmul", memory["output"])
    out_matmul_flex = OutFlexData() if out_matmul.dtype == torch.float32 else precision_config.flex_ctx.out_data
    # Unified mx-scale pointer; when scratchpad exists, prefer its mx buffer
    out_matmul_scale = precision_config.c_mx_scale
    if out_matmul_scale is not None:
        out_matmul_scale = out_matmul_scale.data.view(torch.uint8)
        if has_scratchpad and "mx_c_mx_scale" in memory["scratchpad"]:
            out_matmul_scale = memory["scratchpad"]["mx_c_mx_scale"]
    out_matmul_has_mx = out_matmul_scale is not None and out_matmul.element_size() == 1
    # matrix multiplication
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data_w = tuple([None] * 6) if ragged_dimension != "K" else ragged_metadata_fields(b_ragged_metadata, opt_flags.block_k)
    expt_data_x = tuple([None] * 6) if ragged_dimension is None else ragged_metadata_fields(a_ragged_metadata, opt_flags.block_m if ragged_dimension == "M" else opt_flags.block_k)
    # spmd grid
    grid_m = triton.cdiv(M, opt_flags.block_m)
    if ragged_dimension == "M":
        grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    max_grid = batch_size * grid_m * grid_n * opt_flags.split_k
    grid = min(target_info.num_sms() - opt_flags.idle_sms, max_grid) if opt_flags.is_persistent else max_grid
    # canonicalize storage
    has_scatter_tma = scatter_indx is not None and target_info.has_tma_gather()
    c = wrap_torch_tensor(out_matmul.view(math.prod(out_matmul.shape[:-1]), out_matmul.shape[-1]) if has_scatter else out_matmul.view(math.prod(out_matmul.shape[:-2]), *out_matmul.shape[-2:]))
    a_storage = _canonicalize_storage(a.storage, 2 if has_gather_tma else 3, flex.lhs_data)
    b_storage = _canonicalize_storage(b.storage, 3, flex.rhs_data)
    c_storage = _canonicalize_storage(c.storage, 2 if has_scatter_tma else 3, flex.out_data)
    # create tma descriptor for x
    if c_acc_in is not None:
        assert opt_flags.split_k == 1, "c_acc_in + split_k is not supported."
        assert scatter_indx is None, "c_acc_in + scatter is not supported."
        if c_acc_in.ndim == 2:
            c_acc_in = c_acc_in.unsqueeze(0)
        assert c_acc_in.shape == out_matmul.shape[-3:]
        c_acc_strides = c_acc_in.stride()
    else:
        c_acc_strides = (None, None, None)

    a_tma_block_size = [1, opt_flags.block_k] if has_gather_tma else [1, opt_flags.block_m, opt_flags.block_k]
    a_tma_mode = None if not a_has_tma else "ragged" if ragged_dimension == "M" and not has_gather_tma else "dense"
    a_tensor_or_tma = a_storage.make_tma(a_tma_block_size, a_tma_mode) if a_has_tma else a_storage.data
    # create tma descriptor for y
    c_has_tma = (
        opt_flags.is_persistent and (scatter_indx is None or has_scatter_tma)
        and (c_acc_in is None or c_acc_is_c)
    )
    block_n = opt_flags.block_n // opt_flags.epilogue_subtile // matmul_fused_activation.specs.reduction_n
    c_tma_block_size = [1, block_n] if has_scatter_tma else [1, opt_flags.block_m, block_n]
    c_tma_mode = None if not c_has_tma else "ragged" if is_c_ragged and not has_scatter_tma else "dense"
    c_tensor_or_tma = c_storage.make_tma(c_tma_block_size, c_tma_mode) if c_has_tma else c_storage.data
    # create tma descriptor for w
    b_has_tma = opt_flags.is_persistent
    b_tensor_or_tma = b_storage.make_tma([1, opt_flags.block_k, opt_flags.block_n], "dense") if b_has_tma else b_storage.data
    # create tma descriptor for w_scale
    b_scale_has_tma = opt_flags.is_persistent and b_scale is not None
    b_transpose = b_storage.data.stride()[-2] == 1
    if b_scale_has_tma:
        scale_block_k = opt_flags.block_k // int(MXFP_BLOCK_SIZE)
        b_scale_storage = b_scale.storage
        b_scale_tma_block_size = [scale_block_k, opt_flags.block_n]
        if isinstance(b_scale_storage.layout, (StridedLayout, HopperMXScaleLayout)):
            b_scale_storage = _canonicalize_storage(b_scale.storage, 3, None)
            b_scale_tma_block_size = [1] + b_scale_tma_block_size
        b_scale_tensor_or_tma = b_scale_storage.make_tma(b_scale_tma_block_size, "dense", is_scale=True)
    else:
        b_scale_tensor_or_tma = b_scale
    # create tma descriptor for x_scale
    a_scale_has_tma = False
    if a_has_mx and isinstance(a_scale.storage.layout, BlackwellActMXScaleLayout):
        # check if we can use tma for x scale
        assert opt_flags.is_persistent, "swizzled x scale is only supported for persistent case"
        assert opt_flags.block_m == 128 and opt_flags.block_k >= 128, "block_m and block_k must be at least 128 if x scale is swizzled"
        a_scale_has_tma = True
    if a_scale_has_tma:
        a_scale.storage.data = a_scale.storage.data.view(torch.uint8)
        a_scale.dtype = torch.uint8
        scale_block_k = opt_flags.block_k // int(MXFP_BLOCK_SIZE)
        a_scale_tma_block_size = [opt_flags.block_m, scale_block_k]
        a_scale_tensor_or_tma = a_scale.storage.make_tma(a_scale_tma_block_size, "dense", is_scale=True)
    else:
        a_scale_tensor_or_tma = None if a_scale is None else a_scale.data.view(torch.uint8)
    # canonicalize strides
    a_strides = [0]*(3 - a_storage.data.ndim) + list(a_storage.data.stride())
    a_scale_strides = a_scale.stride() if a_has_mx and not a_scale_has_tma else (None, None, None)
    a_scale_strides = (0, ) * (3 - len(a_scale_strides)) + a_scale_strides
    b_scale_strides = b_scale.stride() if b_has_mx and not b_scale_has_tma else (None, None, None)
    b_scale_strides = (0, ) * (3 - len(b_scale_strides)) + b_scale_strides

    out_matmul_scale_strides = out_matmul_scale.stride() if out_matmul_has_mx else (None, None, None, None)
    out_matmul_scale_strides = (0, ) * (4 - len(out_matmul_scale_strides)) + out_matmul_scale_strides
    # launch kernel
    kernels = specializations.get(epilogue=epilogue.specs, activation=matmul_fused_activation.specs)
    # When stride(-2) == stride(-1) == 1, it's ambiguous whether W is transposed
    # (i.e. col-wise). Since this matters when w_has_mx is True and w_transpose
    # is True the fast code path, stride(-2) == 1 takes precedence, e.g., vs.
    # w_transpose = w_storage.data.stride()[-1] != 1
    fused_comm_kwargs = {
        "pYPtrs": fused_comm.out_handles,
        "ScatterShardIndx": fused_comm.scatter_shard_indx,
        "reduce_rank": fused_comm.reduce_rank,
        "n_reduce_shards": fused_comm.n_reduce_shards,
    } if fused_comm is not None else {}
    n_valid_slices = b_tensor_or_tma.shape[0] if ragged_dimension == "M" else n_slices
    (kernels._p_matmul if opt_flags.is_persistent else kernels._matmul)[(grid,)](
                   c_tensor_or_tma, c_storage.data, *out_matmul.stride(),
                   *((None, out_matmul_scale, None) if out_matmul_has_mx else out_matmul_flex),
                   *out_matmul_scale_strides[-4:],
                   a_tensor_or_tma, a_storage.data, *a_strides, a_transpose,
                   flex.lhs_data.scale,
                   a_scale_tensor_or_tma, *a_scale_strides,
                   b_tensor_or_tma, b_storage.data, *b_storage.data.stride(), b_transpose,
                   flex.rhs_data.scale,
                   b_scale_tensor_or_tma, *b_scale_strides,
                   flex.acc_data.reinterpret(c_acc_in), *c_acc_strides,
                   flex.acc_data.scale, c_acc_is_c,
                   bias, bias_stride,
                   None if ragged_dimension == "M" else a.shape[-2],
                   N, K, K_W,
                   betas, gammas,
                   gather_indx,
                   scatter_indx,
                   None if scatter_indx is None else scatter_indx.shape[0],
                   ragged_dimension,
                   *expt_data_x,
                   *expt_data_w,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   *matmul_fused_activation.fn_args, matmul_fused_activation.specs.reduction_n,
                   *epilogue.fn_arg_values_matmul,
                   n_valid_slices,
                   precision_config.max_num_imprecise_acc,
                   precision_config.allow_tf32,
                   precision_config.flexpoint_saturate_inf,
                   flex.rhs_data.is_per_batch,
                   out_matmul_flex.is_per_batch,
                   flex.acc_data.is_per_batch,
                   opt_flags.block_m,
                   opt_flags.block_n,
                   opt_flags.block_k,
                   opt_flags.group_m,
                   XCD_SWIZZLE=opt_flags.xcd_swizzle,
                   SWIZZLE_MX_VALUE=b.storage.layout.name,
                   SWIZZLE_MX_SCALE=None if b_scale is None else b_scale.storage.layout.name,
                   EPILOGUE_SUBTILE=opt_flags.epilogue_subtile,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=even_K,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=None if a_ragged_metadata is None else a_ragged_metadata.expected_slice_size,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(a, b, out_matmul),
                   X_TMA_MODE=a_tma_mode,
                   Y_TMA_MODE=c_tma_mode,
                   SWAP_XW=get_swap_xw(precision_config, opt_flags),
                   IS_EPILOGUE_QUANT_MXFP8=epilogue.specs.name == FnName.QUANTIZE_MXFP8.name,
                   NUM_SMS = grid if opt_flags.is_persistent else 0,
                   **fused_comm_kwargs,
                   **opt_flags.target_kernel_kwargs)

    assert not (opt_flags.split_k > 1 and scatter_indx is not None)
    out_final_mx_scale = None
    if opt_flags.split_k > 1:
        assert not out_matmul_has_mx
        postprocess_fn1 = ReducePostprocessFn(specs=reduce_fused_activation.specs, fn_args=reduce_fused_activation.fn_args)
        postprocess_fn2 = ReducePostprocessFn(specs=epilogue.specs, fn_args=epilogue.fn_arg_values_finalize)
        c, y_mx_scale = reduce(
            x = out_matmul.view(out_matmul.shape[0], -1, out_matmul.shape[-1]),
            dim = 0,
            # output data/metadata
            y = memory["output"].view(-1, memory["output"].shape[-1]),
            y_dtype = memory["output"].dtype,
            y_flex = precision_config.flex_ctx.out_data,
            y_flex_saturate_inf = precision_config.flexpoint_saturate_inf,
            y_has_mx = precision_config.c_mx_scale is not None,
            # fused functions
            postprocess_fn1 = postprocess_fn1,
            postprocess_fn2 = postprocess_fn2,
        )
        y_shape = out_matmul.shape[1:-1] + (out_matmul.shape[-1] // reduce_fused_activation.specs.reduction_n,)
        out_final = c.view(*y_shape)
        if y_mx_scale is not None:
            out_final_mx_scale = y_mx_scale.view(out_matmul.shape[-2], triton.cdiv(out_matmul.shape[-1], 32))
    else:
        out_final = out_matmul.squeeze(0)
        out_final_mx_scale = out_matmul_scale

    if not (is_input_batched or b_ragged_metadata is not None):
        out_final = out_final.squeeze(0)
    if out_final_mx_scale is not None:
        precision_config.c_mx_scale = out_final_mx_scale
    return out_final

# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------

def apply_precision(x_tri, w_tri, precision_config):
    from .tensor import convert_layout
    from .tensor_details import layout
    from .numerics_details.mxfp import upcast_from_mxfp

    flex_ctx = precision_config.flex_ctx

    def apply(x, scale):
        if scale is None:
            return x.clone()
        return x.float() * scale

    if precision_config.a_mx_scale is not None:
        mx_axis = x_tri.storage.data.ndim -1
        x_tri = convert_layout(x_tri, layout.StridedLayout)
        x_tri_scale = convert_layout(precision_config.a_mx_scale, layout.StridedLayout)
        x_ref = upcast_from_mxfp(x_tri.storage.data, x_tri_scale.storage.data, torch.bfloat16, axis=mx_axis)
    else:
        x_ref = apply(x_tri, flex_ctx.lhs_data.scale)

    if precision_config.b_mx_scale is not None:
        mx_axis = w_tri.storage.data.ndim - 2
        w_tri = convert_layout(w_tri, layout.StridedLayout)
        w_tri_scale = convert_layout(precision_config.b_mx_scale, layout.StridedLayout)
        w_ref = upcast_from_mxfp(w_tri.storage.data, w_tri_scale.storage.data, torch.bfloat16, axis=mx_axis)
    else:
        w_ref = apply(w_tri, flex_ctx.rhs_data.scale)

    return (
        x_ref, w_ref,
    )


def scale(val, scal):
    if scal is None:
        return val
    elif scal.numel() == 1:
        return val / scal
    else:
        assert val.ndim == 3
        return val / scal[:, None, None]

def compute_actual_scale(x, dtype, per_batch_scale=False):
    from triton_kernels.numerics import MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5
    max_finite = {
        torch.float8_e5m2: MAX_FINITE_FLOAT8E5,
        torch.float8_e4m3fn: MAX_FINITE_FLOAT8E4NV,
        torch.float8_e4m3fnuz: MAX_FINITE_FLOAT8E4B8,
    }[dtype]
    maxvals = x.abs().amax(dim=tuple(range(1, x.ndim))) if per_batch_scale else x.abs().max()
    return maxvals / max_finite


def matmul_torch(a, b, bias,
                 a_ragged_metadata: RaggedTensorMetadata | None = None,
                 b_ragged_metadata: RaggedTensorMetadata | None = None,
                 gather_indx: torch.Tensor = None,
                 scatter_indx: torch.Tensor = None,
                 precision_config: PrecisionConfig = None,
                 betas = None,
                 gammas = None,
                 round_x = None, round_y = None,
                 ):
    a, b = apply_precision(a, b, precision_config)

    if b_ragged_metadata is not None:
        n_expts_tot = b_ragged_metadata.slice_sizes.shape[0]
        m, n = a.shape[-2], b.shape[-1]
        out = torch.zeros((n_expts_tot, m, n), dtype=torch.float32, device=a.device)
        x_slice_offs = a_ragged_metadata.slice_offs
        w_slice_offs = b_ragged_metadata.slice_offs
        for expt in range(n_expts_tot):
            k = int(b_ragged_metadata.slice_sizes[expt].item())
            if k == 0:
                continue
            x_start = int(x_slice_offs[expt].item())
            w_start = int(w_slice_offs[expt].item())
            x_slice = a[:, x_start:x_start + k]
            w_slice = b[w_start:w_start + k, :]
            out_expt = matmul_torch(
                x_slice, w_slice, None, None,
                None, None, None, PrecisionConfig(),
                betas, gammas,
                round_x, round_y,
            )
            out[expt] = out_expt.to(out.dtype)
        actual_scale = precision_config.flex_ctx.out_data.actual_scale
        if actual_scale is not None:
            actual_scale.copy_(compute_actual_scale(out, precision_config.out_dtype))
        return scale(out, precision_config.flex_ctx.out_data.expected_scale)

    is_input_batched = a.ndim == 3
    assert a.dtype.itemsize > 1
    assert b.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert b.ndim == 3 and b.shape[0] == a.shape[0]
    if round_x is None:
        round_x = lambda x, idx: x
    if round_y is None:
        round_y = lambda x: x
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if b.ndim == 2:
        b = b.view(1, *b.shape)
    if a.ndim == 2:
        a = a.view(1, *a.shape)
    # memory offsets
    if a_ragged_metadata is not None and not is_input_batched:
        sizes = a_ragged_metadata.slice_sizes
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, a.shape[1]] for _ in range(b.shape[0])]
    # compute
    n_rows = a.shape[1] if gather_indx is None else gather_indx.shape[0]
    y = torch.zeros((a.shape[0], n_rows, b.shape[-1]), device=a.device, dtype=a.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=a.device)
        else:
            idx = gather_indx[lo:hi]
        batch = i if is_input_batched else 0
        out = torch.matmul(round_x(a[batch, idx, :], torch.arange(lo, hi, device="cuda")).float(),
                           b[i].float())
        if bias is not None:
            out += bias[i, :] if betas is None else bias[i, :] * betas[lo:hi, None]
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[batch, lo:hi, :] = round_y(out)
    if not is_input_batched:
        y = y.view(y.shape[1], y.shape[2])
    if scatter_indx is None:
        out = y
    else:
        out = torch.zeros((scatter_indx.shape[0], y.shape[-1]), dtype=y.dtype, device=a.device)
        msk = scatter_indx != -1
        out[scatter_indx[msk], :] = y[msk, :]
    actual_scale = precision_config.flex_ctx.out_data.actual_scale
    if actual_scale is not None:
        actual_scale.copy_(compute_actual_scale(out, precision_config.out_dtype))
    return scale(out, precision_config.flex_ctx.out_data.expected_scale)


def post_matmul_comm_torch(y: torch.Tensor, rank: int, n_reduce_shards: int,
                           world_size: int,
                           scatter_shard_indx: torch.Tensor | None = None,
):
    """
    Reference implementation of post matmul communication.

    y: the local matmul output
    rank: the global rank
    n_reduce_shards: the number of reduce shards
    world_size: the world size
    scatter_shard_indx: the shard indices for the scatter. None if all gather.

    Output shape:
    (batch_size, n_rows, n_cols) -> (batch_size, n_rows * n_reduce_shards, n_cols) if batched, otherwise
    (n_rows, n_cols) -> (n_rows * n_reduce_shards, n_cols)
    """
    from torch import distributed as dist
    # if n_reduce_shards == 1:
    #     return y

    ys = [torch.empty_like(y) for _ in range(world_size)]
    dist.all_gather(ys, y)
    out_shape = (*y.shape[:-2], y.shape[-2] * n_reduce_shards, y.shape[-1])

    if scatter_shard_indx is None:
        # all gather
        assert n_reduce_shards == world_size
        return torch.cat(ys, dim=-1).reshape(out_shape)
    else:
        # Note: when multiple ranks scatter to the same destination, the result is undefined.
        scatter_shard_indx_global = torch.empty((world_size, *scatter_shard_indx.shape), device=scatter_shard_indx.device, dtype=scatter_shard_indx.dtype)
        dist.all_gather([scatter_shard_indx_global[i] for i in range(world_size)], scatter_shard_indx)

        assert len(out_shape) == 2, "batched mode not supported"
        result = torch.zeros(out_shape, device=y.device, dtype=y.dtype)
        reduce_shard_id = rank // n_reduce_shards

        for i in range(world_size // n_reduce_shards):
            scatter_mask = scatter_shard_indx_global[i * n_reduce_shards, :] == reduce_shard_id
            for j in range(n_reduce_shards):
                out_slice = result.as_strided(
                    (result.shape[0] // n_reduce_shards, result.shape[1]),
                    (result.stride(0) * n_reduce_shards, result.stride(1)),
                    storage_offset=j * result.stride(0),
                )
                out_slice[scatter_mask, :] = ys[i * n_reduce_shards + j][scatter_mask, :]
        return result
