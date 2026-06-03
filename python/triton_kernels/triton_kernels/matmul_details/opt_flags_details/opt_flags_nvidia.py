import torch
import triton
from triton_kernels import target_info
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE
from triton_kernels.tensor import FP4, FP16, FP32, BF16, Tensor
from triton_kernels.tensor_details.layout import HopperMXScaleLayout
from triton_kernels.tensor_details.layout_details.blackwell_scale import BlackwellActMXScaleLayout, BlackwellMXScaleLayout


def is_x_scale_swizzled(precision_config):
    return (precision_config is not None and precision_config.a_mx_scale is not None
            and isinstance(precision_config.a_mx_scale, Tensor)
            and isinstance(precision_config.a_mx_scale.storage.layout, BlackwellActMXScaleLayout))


def compute_swap_xw(precision_config, block_m, is_persistent, lhs_dtype, rhs_dtype):
    if target_info.cuda_capability_geq(10, 0):
        if lhs_dtype == FP4 and rhs_dtype == FP4:
            return block_m <= 128 and is_persistent
        if precision_config.b_mx_scale is not None:
            return block_m <= 64 and is_persistent
        else:
            return block_m < 64 and is_persistent
    elif target_info.cuda_capability_geq(9, 0):
        layout = None if not isinstance(precision_config.b_mx_scale,
                                        Tensor) else precision_config.b_mx_scale.storage.layout
        return isinstance(layout, HopperMXScaleLayout)

    return False


def compute_grid_size(routing_data, batch_size, m, n, block_m, block_n):
    if routing_data is not None and batch_size == 1:
        grid_m = routing_data.n_blocks(routing_data.n_slices, m, block_m)
    else:
        grid_m = triton.cdiv(m, block_m)
    grid_n = (n + block_n - 1) // block_n
    return batch_size * grid_m * grid_n


def compute_block_n(n: int, arch, precision_config):
    # block_n:
    layout = None if not isinstance(precision_config.b_mx_scale, Tensor) else precision_config.b_mx_scale.storage.layout
    if isinstance(layout, HopperMXScaleLayout):
        # https://github.com/triton-lang/triton/blob/814b862166c756d9f33238844f4ac047e0243388/python/triton_kernels/triton_kernels/matmul_details/_matmul.py#L265
        block_n = 2 * layout.num_warps * 2 * 8
        return block_n, block_n
    if precision_config.max_num_imprecise_acc is None and n > 128:
        block_n, block_n_tma = 256, 256
    else:
        target = min(128, triton.next_power_of_2(n))
        block_n, block_n_tma = max(8, target), max(16, target)
    if isinstance(layout, BlackwellMXScaleLayout):
        # Blackwell scale swizzle requires BLOCK_N to be a multiple of 128.
        block_n = max(128, block_n)
        block_n_tma = max(128, block_n_tma)
    return block_n, block_n_tma


def compute_block_k(m: int, k: int | None, is_persistent: bool, lhs_dtype, rhs_dtype, precision_config, has_y_acc_in):
    lhs_width = lhs_dtype.bitwidth
    rhs_width = rhs_dtype.bitwidth
    # block_k needs to match the cacheline size (1024 bits)
    block_k = int(1024 // min(lhs_width, rhs_width))
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if rhs_width == 4 and not has_native_mxfp:
        block_k = 128
    elif is_persistent and is_x_scale_swizzled(precision_config):
        # x scale has been swizzled to BlackwellActMXScaleLayout, enforce block_k to be multiple of 128
        block_k = max(block_k, 128)
    elif k is not None:  # cover small k case
        min_block_k = 32 if is_persistent or lhs_width != 16 or rhs_width != 16 else 16
        block_k = max(min_block_k, min(triton.next_power_of_2(k), block_k))
    has_mx_weight_scale = precision_config is not None and precision_config.b_mx_scale is not None
    if has_native_mxfp and is_persistent and has_mx_weight_scale:
        # If both inputs are fp4, allow larger block_k.
        max_block_k = 256 if lhs_dtype == FP4 and rhs_dtype == FP4 else 128
        block_k = min(block_k, max_block_k)
    if has_y_acc_in and lhs_width == rhs_width == 16 and not target_info.cuda_capability_geq(10, 0):
        block_k = min(block_k, 32)
    return block_k


def compute_split_k(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = triton.cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


def compute_num_warps(block_m, block_n, is_persistent: bool, precision_config, constraints):
    layout = None if not isinstance(precision_config.b_mx_scale, Tensor) else precision_config.b_mx_scale.storage.layout
    if isinstance(layout, HopperMXScaleLayout):
        return layout.num_warps
    num_warps = constraints.get("num_warps", None)
    if num_warps is not None:
        return num_warps
    return max(block_m * block_n // 4096, 4 if is_persistent else 1)


def compute_num_stages(
    precision_config,
    is_persistent,
    block_m,
    block_n,
    block_k,
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    x_transpose,
    epilogue_effective_itemsize,
    has_y_acc_in,
    mx_block_size=None,
    epilogue_reduction_n=1,
    *,
    epilogue_subtile,
    occupancy_target,
):
    if precision_config.max_num_imprecise_acc is not None:
        return 3
    act_size = lhs_dtype.bitwidth / 8
    weight_size = rhs_dtype.bitwidth / 8
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if has_native_mxfp and precision_config.b_mx_scale is not None and lhs_dtype in [FP16, BF16]:
        # For fp16/bf16 x mxfp, we upcast weight on the fly, so size
        # smem_capacity accordingly.
        # w/o this, gets the following error:
        # "triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 263356, Hardware limit: 232448. Reducing block sizes or `num_stages` may help"
        # for x.shape = [2048, >=4096] bf16 x [32, >=4096, >=4096] float8_e4m3fn
        # block_m=64, block_n=256, block_k=128, split_k=1, is_persistent=True -> leading to num_stages=4
        weight_size = 2

    stage_size = block_m * block_k * act_size + block_k * block_n * weight_size
    device_props = torch.cuda.get_device_properties(0)
    smem_capacity = device_props.shared_memory_per_block_optin
    smem_capacity //= occupancy_target
    if has_native_mxfp:
        # 4-bit e2m1 operands are padded 2x
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
        if precision_config.a_mx_scale is not None and lhs_dtype == FP4 and rhs_dtype != FP4:
            stage_size += block_k * block_n * act_size
        if precision_config.b_mx_scale is not None and rhs_dtype == FP4 and lhs_dtype != FP4:
            stage_size += block_k * block_n * weight_size

    if precision_config.a_mx_scale is not None:
        scale_block_size = mx_block_size or int(MXFP_BLOCK_SIZE)
        stage_size += block_m * (block_k // scale_block_size)

    if precision_config.b_mx_scale is not None:
        # mx scales
        scale_block_size = mx_block_size or int(MXFP_BLOCK_SIZE)
        stage_size += block_n * (block_k // scale_block_size)

    if is_persistent:
        # Per-stage wait barrier
        stage_size += 8
        out_itemsize = (out_dtype.bitwidth / 8) * (1.25 if has_y_acc_in else 1.0)
        if target_info.cuda_capability_geq(10, 0):
            acc_size = epilogue_effective_itemsize or out_itemsize
        else:
            acc_size = out_itemsize
        if target_info.cuda_capability_geq(10, 0) and epilogue_subtile is not None:
            acc_block_n = block_n // epilogue_subtile // epilogue_reduction_n
        else:
            acc_block_n = block_n // epilogue_reduction_n
        # pipelined TMA store local to global, or
        # pipelined layout conversion before store of the accumulator
        # note: layout conversion has some padding
        epilogue_smem = int((block_m + 4) * acc_block_n * acc_size)
        if compute_swap_xw(precision_config, block_m, is_persistent, lhs_dtype, rhs_dtype):
            # The fp32 accumulator stays in TMEM for the Blackwell SWAP_XW
            # persistent path. Fused reductions such as swiglu still need smem
            # for the unreduced output tile before the narrower TMA-store tile.
            if epilogue_reduction_n > 1 or epilogue_subtile > 1:
                epilogue_smem += int(block_m * block_n * out_itemsize)
        smem_capacity -= epilogue_smem
        if x_transpose:
            smem_capacity -= int(block_m * block_k * act_size)

    # Persistent fp32 kernels need extra smem headroom (metadata/barriers/TMA state)
    # that is not fully captured by the simple stage_size model above.
    if is_persistent and (lhs_dtype == FP32 or rhs_dtype == FP32):
        smem_capacity -= 32 * 1024
    if is_persistent and not has_native_mxfp and epilogue_reduction_n > 1:
        # Hopper fused reductions materialize an additional reduced-N output
        # tile in smem.
        smem_capacity -= int(block_m * acc_block_n * out_itemsize)
    smem_capacity = max(smem_capacity, 0)
    max_stages = 5 if rhs_dtype == FP4 else 4  # maybe 5 everywhere; just haven't tested
    b_mx_scale_layout = None if not isinstance(precision_config.b_mx_scale,
                                               Tensor) else precision_config.b_mx_scale.storage.layout
    if (is_persistent and rhs_dtype == FP4 and isinstance(b_mx_scale_layout, HopperMXScaleLayout)
            and precision_config.a_mx_scale is not None and precision_config.c_mx_scale is not None):
        # The Hopper-scale FP4 path with MX input and output needs enough
        # extra epilogue/scale smem that a 5-stage persistent kernel can
        # exceed H100's launch limit.
        max_stages = 4
    num_stages = min(smem_capacity // int(stage_size), max_stages)
    # Keep one stage of headroom for persistent fp32 to avoid launch-time OOR.
    if is_persistent and (lhs_dtype == FP32 or rhs_dtype == FP32):
        num_stages = min(num_stages, 3)
    if num_stages == 0:
        num_stages = 1
    return num_stages
