import torch
import triton
from triton_kernels import target_info


def compute_grid_size(routing_data, m, n, block_m, block_n):
    if routing_data is not None:
        grid_m = routing_data.n_blocks(m, block_m)
    else:
        grid_m = triton.cdiv(m, block_m)
    grid_n = (n + block_n - 1) // block_n
    return grid_m * grid_n


def compute_block_n(n: int, arch, precision_config):
    capability = torch.cuda.get_device_capability()[0] if arch is None else int(arch[2:-1])
    # block_n:
    block_n = max(16, min(128, triton.next_power_of_2(n)))
    if capability >= 9 and precision_config.max_num_imprecise_acc is None and n > 128:
        block_n = 256
    return block_n


def compute_block_k(k: int | None, is_persistent: bool, lhs_dtype, rhs_dtype, mx_ctx):
    has_mx_weight_scale = mx_ctx and mx_ctx.weight_scale is not None
    lhs_width = lhs_dtype.itemsize
    rhs_width = rhs_dtype.itemsize
    if has_mx_weight_scale:
        rhs_width = 0.5
    # block_k needs to match the cacheline size (128B)
    block_k = int(128 // min(lhs_width, rhs_width))
    # TODO: revisit when Triton is better for H100 + MXFP4
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if rhs_width == 0.5 and not has_native_mxfp:
        block_k = 128
    elif k is not None:
        block_k = max(32, min(triton.next_power_of_2(k), block_k))

    if has_native_mxfp and is_persistent and has_mx_weight_scale:
        # Cap block_k to conserve smem to increase num_stages
        block_k = min(block_k, 128)
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


def compute_num_warps(block_m, block_n):
    return max(block_m * block_n // 4096, 4)


def compute_num_stages(
    precision_config,
    microscaling_ctx,
    is_persistent,
    block_m,
    block_n,
    block_k,
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    epilogue_subtile,
    has_expensive_epilogue,
):
    if precision_config.max_num_imprecise_acc is not None:
        return 3
    weight_size = 0.5 if rhs_dtype == torch.uint8 else rhs_dtype.itemsize
    stage_size = block_m * block_k * lhs_dtype.itemsize + block_k * block_n * weight_size
    device_props = torch.cuda.get_device_properties(0)
    smem_capacity = device_props.shared_memory_per_block_optin
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if has_native_mxfp and microscaling_ctx is not None:
        if microscaling_ctx.weight_scale is not None:
            if rhs_dtype == torch.uint8:
                # 4-bit e2m1 weights are padded 2x
                # https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
                stage_size += block_k * block_n * weight_size

    if is_persistent:
        # Per-stage wait barrier
        stage_size += 8
        acc_size = out_dtype.itemsize
        if target_info.cuda_capability_geq(10, 0):
            acc_size = 4 if has_expensive_epilogue else out_dtype.itemsize
        else:
            acc_size = out_dtype.itemsize
        if target_info.cuda_capability_geq(10, 0) and epilogue_subtile and not has_expensive_epilogue:
            acc_block_n = block_n // 2
        else:
            acc_block_n = block_n
        # pipelined TMA store local to global, or
        # pipelined layout conversion before store of the accumulator
        # note: layout conversion has some padding
        smem_capacity -= (block_m + 4) * acc_block_n * acc_size
        if microscaling_ctx.weight_scale is not None:
            # mx scales
            stage_size += block_n * (block_k // 32)
    elif has_native_mxfp:
        # mx scales
        stage_size += block_n * (block_k // 32)
    num_stages = min(4, smem_capacity // int(stage_size))
    return num_stages
