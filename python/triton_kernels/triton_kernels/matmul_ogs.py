from dataclasses import dataclass
import itertools
import math
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
from .matmul_ogs_details.opt_flags import make_opt_flags, OptFlags
from .matmul_ogs_details.fast_contiguous import fast_contiguous
from .numerics_details.mxfp import SwizzlingType
from .specialize import specialize
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
    def create_basic_descriptor(tensor: torch.Tensor, block_shape: Tuple[int, ...],
                                transpose: bool = False) -> TensorDescriptor:
        """Create a basic tensor descriptor with optional transpose"""
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            tensor = tensor.permute(0, 2, 1)
        return TensorDescriptor.from_tensor(tensor, block_shape=block_shape)

    @staticmethod
    def create_weight_descriptor(w_tensor: torch.Tensor, block_k: int, block_n: int,
                                 transpose: bool) -> TensorDescriptor:
        """Create a tensor descriptor for weight matrix"""
        # Two e2m1 packed in a uint8 or a single fp8
        W_PACK_DIVISOR = 2 if w_tensor.dtype == torch.uint8 else 1
        PACKED_BLOCK_K_W = block_k // W_PACK_DIVISOR
        return TensorDescriptorBuilder.create_basic_descriptor(w_tensor, block_shape=[1, PACKED_BLOCK_K_W, block_n],
                                                               transpose=transpose)

    @staticmethod
    def create_block_scale_descriptor(mx_tensor: torch.Tensor, block_k: int, block_n: int, K: int, N: int,
                                      mx_scale_stride_k: int, mx_scale_stride_n: int, n_expts_tot: int, batch_size: int,
                                      expt_data: Optional[ExptData], swizzle_mx: bool,
                                      transpose: bool) -> TensorDescriptor:
        """Create a tensor descriptor for block scale factors"""
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_k // MX_PACK_DIVISOR
        PackedK = (K + MX_PACK_DIVISOR - 1) // MX_PACK_DIVISOR

        if swizzle_mx:
            num_expt_x_ncol = (n_expts_tot if expt_data is not None and len(expt_data.block_pid_map) > 0 else
                               batch_size) * ((N + 127) // 128)
            return TensorDescriptor(
                base=mx_tensor, shape=[1, num_expt_x_ncol, (PackedK + 3) // 4, 2, 256],
                strides=[num_expt_x_ncol * mx_scale_stride_n, mx_scale_stride_n, mx_scale_stride_k, 256,
                         1], block_shape=[1, block_n // 128, MX_SCALE_BLOCK_K // 4, 2, 256])
        else:
            # Non-optimal SF layout, expect slow transfers
            # from global to shmem and from shmem to tmem
            return TensorDescriptorBuilder.create_basic_descriptor(mx_tensor,
                                                                   block_shape=[1, MX_SCALE_BLOCK_K,
                                                                                block_n], transpose=transpose)

    @staticmethod
    def squeeze_after_dim(x, dim=2):
        shape = list(x.shape)
        new_shape = [s for s in shape[:dim - 1] if s != 1] + shape[dim - 1:]
        return x.view(*new_shape)

    @staticmethod
    def create_input_descriptor_gather(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int,
                                       block_k: int) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X via TMA gather"""
        x_desc = TensorDescriptorBuilder.squeeze_after_dim(x_tensor)
        assert x_desc.ndim == 2, "TMA gather descriptor requires 2D input"
        INT_MAX = 2147483647
        return TensorDescriptor(base=x_desc, shape=[INT_MAX, K], strides=[x_stride_1, x_stride_2],
                                block_shape=[1, block_k])

    @staticmethod
    def create_input_descriptor_load(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int, block_m: int,
                                     block_k: int) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X via TMA"""
        x_desc = TensorDescriptorBuilder.squeeze_after_dim(x_tensor)
        assert x_desc.ndim in [2, 3], "LHS input TMA descriptor builder expects 2D or 3D input"
        return TensorDescriptor(base=x_desc, shape=[x_desc.shape[0], K], strides=[x_stride_1, x_stride_2],
                                block_shape=[block_m, block_k])

    @staticmethod
    def create_input_descriptor(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int, block_k: int,
                                block_m: int, use_gather_tma: bool, use_load_tma: bool) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X based on TMA usage"""
        if use_gather_tma:
            return TensorDescriptorBuilder.create_input_descriptor_gather(x_tensor, K, x_stride_1, x_stride_2, block_k)
        elif use_load_tma:
            return TensorDescriptorBuilder.create_input_descriptor_load(x_tensor, K, x_stride_1, x_stride_2, block_m,
                                                                        block_k)
        else:
            return x_tensor


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

    def check_inputs(self, weights: torch.Tensor) -> None:
        if self.swizzle_value and self.swizzle_value != SwizzlingType.HOPPER:
            raise ValueError(f"Only Hopper swizzling is supported. Got {self.swizzle_value}")

        if self.weight_scale is None:
            return

        valid_weight_types = {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}
        # Validate weights data type
        if weights.dtype not in valid_weight_types:
            raise TypeError(f"Weights must be one of {valid_weight_types}. Got {weights.dtype}")

        # Validate weights tensor dimensions
        if weights.ndim != 3:
            raise ValueError(f"Weights must be 3D (experts, in_dim, out_dim). Got {weights.shape}")
        weights_shape = self.get_packed_tensor_logical_shape(weights)

        # Validate shapes
        weight_scale_shape = self.actual_weight_scale_shape
        if weights_shape[0] != weight_scale_shape[0] or weights_shape[2] != weight_scale_shape[2]:
            raise ValueError(
                f"Weights and scale must have the same number of experts and output dimensions. "
                f"Got weights experts: {weights_shape[0]}, scale experts: {weight_scale_shape[0]}, "
                f"weights out_dim: {weights_shape[2]}, scale out_dim: {weight_scale_shape[2]}"
            )
        if self.swizzle_value == SwizzlingType.HOPPER:
            if weights_shape[1] % 64 != 0 or weights_shape[2] % 16 != 0:
                raise ValueError(
                    f"Hopper scale swizzling acts on a 64x16 tile (4x1 mma tiles). Got {weights_shape=}"
                )
        if self.swizzle_scale == SwizzlingType.HOPPER:
            if weights_shape[1] % 64 != 0 or weights_shape[2] % 64 != 0:
                raise ValueError(
                    f"Hopper scale swizzling acts on a 64x64 tile (4x4 mma tiles). Got {weights_shape=}"
                )

        k_dim = weights_shape[1]
        rounded_k_dim = (k_dim + 31) // 32 * 32
        block_size = rounded_k_dim // weight_scale_shape[1]
        if block_size != 32:
            raise ValueError(f"Block size must be 32. Got {block_size}")

    def compute_strides(self):
        if self.weight_scale is not None:
            # Check expected properties of the weights.
            if self.swizzle_scale == SwizzlingType.BLACKWELL:
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
        shape = list(tensor.shape)
        if tensor.dtype == torch.uint8:
            # Assume 2 fp4s packed into a byte
            shape[1] *= 2
        if self.swizzle_value == SwizzlingType.HOPPER:
            shape[1] //= 4
            shape[2] *= 4
        return tuple(shape)

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
        assert self.flex_ctx.rhs_data.scale is None or self.mx_ctx.weight_scale is None, "flex and mx_ctx cannot be used together"

def mx_can_use_tma(mx_ctx: MicroscalingCtx):
    mx_scale_stride_e, mx_scale_stride_n, mx_scale_stride_k = mx_ctx.compute_strides()
    if mx_scale_stride_e * mx_ctx.weight_scale.element_size() % 16 != 0:
        return False

    if mx_ctx.swizzle_scale == SwizzlingType.BLACKWELL:
        # Check stride in bytes are multiples of 16.
        return mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0
    elif mx_ctx.swizzle_scale is None:
        # Check MX is either transposed or non-transposed, and with required stride.
        return (
            (mx_scale_stride_n * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_k == 1) or
            (mx_scale_stride_k * mx_ctx.weight_scale.element_size() % 16 == 0 and mx_scale_stride_n == 1)
        )
    else:
        return False

def can_use_persistent_tma(x, w, gather_indx, precision_config):
    mx_ctx = precision_config.mx_ctx
    is_mxfp4 = mx_ctx.weight_scale is not None and w.dtype == torch.uint8
    weight_stride_req = 32 if is_mxfp4 else 16
    return (
        # TMA requires CUDA 9.0, last dim contiguous, and multiple of 16-byte strides otherwise.
        target_info.cuda_capability_geq(9, 0) and
        (True if gather_indx is not None else
            # Check strides of X.
            x.stride(1) * x.element_size() % 16 == 0 and x.stride(2) == 1
        ) and (
            # Check W is either transposed or non-transposed, and with required stride.
            (w.stride(1) * w.element_size() % weight_stride_req == 0 and w.stride(2) == 1) or
            (w.stride(2) * w.element_size() % weight_stride_req == 0 and w.stride(1) == 1)
        ) and (
            mx_ctx.weight_scale is None or mx_can_use_tma(mx_ctx)
        )
        # compiler crash ?
        and (x.dtype.itemsize <= 1 or w.dtype != torch.uint8)
        # NYI
        and mx_ctx.swizzle_value is None
    )

def can_use_fused_scatter(scatter_indx, fused_activation):
    return scatter_indx is not None and fused_activation.specs.fn is None

# ---------------------
# Preprocessing
# ---------------------

@dataclass(frozen=True)
class PreprocessingFeatures:
    w_want_k_major: bool
    swap_xw: bool

def init_preprocessing_features(w, precision_config, opt_flags):
    mx_ctx = precision_config.mx_ctx
    swap_xw = False  # Whether or not to swap X and W operands to the tl.dot
    w_want_k_major = False
    if not target_info.cuda_capability_geq(10, 0):
        # Hopper transpose. Reduction dimension must be contiguous.
        if w.stride(1) != 1 and w.dtype.itemsize == 1:
            w_want_k_major = True

    if target_info.cuda_capability_geq(10, 0):
        swap_xw = mx_ctx.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
        if swap_xw:
            w_want_k_major = True
    return PreprocessingFeatures(w_want_k_major, swap_xw)


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
    if preprocessing_features.w_want_k_major:
        w = fast_contiguous(w.transpose(-1, -2)).transpose(-1, -2)
    # preprocess routing information and ptr lookup table
    M = x.shape[1] if gather_indx is None else gather_indx.src_indx.shape[0]
    return x, w, preprocessing_features.swap_xw, writeback_idxs, writeback_size, finalize_scatter_idxs


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
    batch_size: int,
    K: int,
    N: int,
    mx_scale_stride_k: int,
    mx_scale_stride_n: int,
    USE_GATHER_TMA: bool,
    X_USE_LOAD_TMA: bool,
    w_transpose: bool,
    mx_transpose: bool,
) -> Tuple[bool, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Create and cache TMA descriptors for tensors."""
    use_host_tma_descriptors = opt_flags.is_persistent and target_info.cuda_capability_geq(10, 0)

    x_desc, w_desc = [None] * 2
    descriptors = []
    # The dense case currently uses on device descriptor updates
    # so we bail out on using host descriptors in that case
    if (use_host_tma_descriptors):
        if USE_GATHER_TMA or X_USE_LOAD_TMA:
            x_desc = TensorDescriptorBuilder.create_input_descriptor(
                    x, K, x.stride(1), x.stride(2),
                    opt_flags.block_k, opt_flags.block_m,
                    USE_GATHER_TMA, X_USE_LOAD_TMA
                )
        descriptors.append(x_desc)
        if (expt_data is not None and len(expt_data.block_pid_map) > 0):
            w_desc = TensorDescriptorBuilder.create_weight_descriptor(
                    w, opt_flags.block_k, opt_flags.block_n, w_transpose
                )
            is_microscaled_format = (mx_ctx.weight_scale is not None) and (w.dtype == torch.uint8)
            if is_microscaled_format:
                # Pad the inner shape to 128 for mxfp4 weights
                # for mixed precision fp8 x mxfp4 compute
                pad = 128
                dim_to_pad = -1
                old_size = w_desc.shape[dim_to_pad]
                padded_size = math.ceil(old_size / pad) * pad
                if padded_size != old_size:
                    w_desc.shape = list(w_desc.shape)
                    w_desc.shape[dim_to_pad] = padded_size
        descriptors.append(w_desc)
        # Optional MX scale descriptor
        descriptors.append(None)
        if mx_tensor is not None:
            descriptors[-1] = TensorDescriptorBuilder.create_block_scale_descriptor(
                    mx_tensor, opt_flags.block_k, opt_flags.block_n, K, N,
                    mx_scale_stride_k, mx_scale_stride_n, routing_data.n_expts_tot,
                    batch_size,
                    expt_data, mx_ctx.swizzle_scale, mx_transpose
                )

    # TODO: Currently all or none, instead should support a mixture
    # of host and device descriptors
    if None in descriptors or len(descriptors) == 0:
        descriptors = [x, w, mx_tensor]
        use_host_tma_descriptors = False
    if opt_flags.is_persistent:
        opt_flags.target_kernel_kwargs["USE_HOST_TMA_DESCRIPTORS"] = use_host_tma_descriptors

    return use_host_tma_descriptors, *descriptors


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
        can_use_fused_scatter(scatter_indx, fused_activation),
        epilogue.effective_itemsize,
    )
    # compute grid size
    if not is_input_batched:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    else:
        grid_m = triton.cdiv(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    assert n_expts_tot == routing_data.n_expts_tot
    assert grid_m > 0
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
    x, w, swap_xw, writeback_idxs, writeback_size, finalize_scatter_idxs = apply_preprocessing_features(
        x, w, gather_indx, scatter_indx, routing_data, opt_flags, preprocessing_features
    )
    # matrix multiplication
    n_cta = batch_size * grid_m * grid_n * opt_flags.split_k
    n_cta = min(target_info.num_sms(), n_cta) if opt_flags.is_persistent else n_cta
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]

    kernels = get_kernels(epilogue.specs, fused_activation.specs)
    expt_data = routing_data.expt_data
    block_m = opt_flags.block_m
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map[block_m]

    HAS_TMA_GS = target_info.cuda_capability_geq(10, 0)
    USE_GATHER_TMA = HAS_TMA_GS and gather_indx is not None
    X_USE_LOAD_TMA = gather_indx is None and not USE_GATHER_TMA
    _, x_tensor, w_tensor, mx_tensor = _create_tma_descriptors(
        x=x, w=w,
        mx_tensor=mx_ctx.weight_scale,
        routing_data=routing_data,
        mx_ctx=mx_ctx,
        expt_data=expt_data,
        opt_flags=opt_flags,
        batch_size=batch_size,
        K=K,
        N=N,
        mx_scale_stride_k=mx_scale_stride_k,
        mx_scale_stride_n=mx_scale_stride_n,
        USE_GATHER_TMA=USE_GATHER_TMA,
        X_USE_LOAD_TMA=X_USE_LOAD_TMA,
        w_transpose=w.stride(2) != 1,
        mx_transpose=mx_scale_stride_n != 1,
    )
    if isinstance(x_tensor, torch.Tensor):
        x_tensor = flex.lhs_data.reinterpret(x)
    if isinstance(w_tensor, torch.Tensor):
        w_tensor = flex.rhs_data.reinterpret(w)
    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(n_cta,)](
                   flex.out_data.reinterpret(memory["output"]),
                   flex.out_data.reinterpret(out0), *out0.stride(),
                   *out0_flex,
                   x_tensor, x.stride(0), x.stride(1), x.stride(2),
                   flex.lhs_data.scale,
                   w_tensor, w.stride(0), w.stride(1), w.stride(2), w.stride(2) != 1,
                   flex.rhs_data.scale,
                   mx_tensor, mx_scale_stride_e, mx_scale_stride_k, mx_scale_stride_n, mx_scale_stride_n != 1,
                   bias, bias_stride,
                   x.shape[1],
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
                   SWAP_XW=swap_xw,
                   NUM_SMS = n_cta if opt_flags.is_persistent else 0,
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
