from dataclasses import dataclass
import triton
from triton_bench import target_info
import torch

from . import opt_flags_amd, opt_flags_nvidia

# fmt: off

@dataclass
class OptFlags:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int
    xcd_swizzle: int
    w_cache_modifier: str
    split_k: int
    fused_scatter: bool
    is_persistent: bool
    epilogue_subtile: bool
    arch: str
    target_kernel_kwargs: dict

    def __post_init__(self):
        if self.fused_scatter and self.split_k != 1:
            raise ValueError("Not supported")



def make_default_opt_flags_amd(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    microscaling_ctx,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
    enforce_bitwise_invariance,
    constraints,
):
    assert not constraints, "flags constraints not supported on AMD"
    # tokens per expert
    if routing_data is None:
        tokens_per_expt = m
    elif routing_data.expected_tokens_per_expt is None:
        tokens_per_expt = max(1, m // routing_data.n_expts_tot)
    else:
        tokens_per_expt = routing_data.expected_tokens_per_expt
    # block_m
    if constraints.get("block_m", None):
        block_m = constraints["block_m"]
    elif enforce_bitwise_invariance:
        block_m = 128
    elif tokens_per_expt >= 512 and n >= 2048:
        block_m = 128
    else:
        block_m = max(32, min(triton.next_power_of_2(tokens_per_expt), 64))
    if routing_data is not None:
        grid_m = routing_data.n_blocks(m, block_m)
    else:
        grid_m = triton.cdiv(m, block_m)
    # group_m:
    group_m = 4
    # number of xcds
    num_xcds = 8
    xcd_swizzle = num_xcds
    # block_nk:
    block_n, block_k = opt_flags_amd.compute_block_nk(
        n, block_m, grid_m, num_xcds, lhs_dtype, rhs_dtype, microscaling_ctx
    )
    # split_k:
    grid_size = grid_m * ((n + block_n - 1) // block_n)
    n_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if enforce_bitwise_invariance:
        split_k = 1
    else:
        split_k = max(1, n_cu // grid_size)
    # w_cache_modifier:
    w_cache_modifier = ".cg" if block_m <= 32 else None
    # num_warps, num_stages
    num_warps = 2 if (m is not None and m <= 16) else 8
    num_stages = 2
    is_persistent = False
    # AMD-specific
    target_kernel_kwargs = {"waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    return OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        w_cache_modifier=w_cache_modifier,
        split_k=split_k,
        fused_scatter=False,
        is_persistent=is_persistent,
        epilogue_subtile=False,
        arch=None,
        target_kernel_kwargs=target_kernel_kwargs,
    )

def make_default_opt_flags_nvidia(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    microscaling_ctx,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
    enforce_bitwise_invariance,
    constraints,
):
    constraints_supported = ["block_m", "block_k", "split_k", "fused_scatter", "is_persistent", "epilogue_subtile"]
    assert not any([c not in constraints_supported for c in constraints]), constraints.keys()
    # tokens per expert
    if routing_data is None:
        tokens_per_expt = m
    elif routing_data.expected_tokens_per_expt is None:
        tokens_per_expt = max(1, m // routing_data.n_expts_tot)
    else:
        tokens_per_expt = routing_data.expected_tokens_per_expt
    # pid swizzling
    group_m = 8
    xcd_swizzle = 1
    # block_m
    if constraints.get("block_m", None):
        block_m = constraints["block_m"]
    elif enforce_bitwise_invariance:
        block_m = 128
    else:
        block_m = max(64, min(triton.next_power_of_2(tokens_per_expt), 128))
    # TODO: remove when triton is more optimized for H100 MXFP4
    arch = None
    if (
        block_m < 128
        and rhs_dtype == torch.uint8
        and microscaling_ctx.weight_scale is not None
        and not target_info.cuda_capability_geq(10, 0)
    ):
        arch = "sm80"
    # block n
    block_n = opt_flags_nvidia.compute_block_n(n, arch, precision_config)
    # is_persistent
    grid_size = opt_flags_nvidia.compute_grid_size(routing_data, m, n, block_m, block_n)
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    tiles_per_sm = grid_size / n_sms
    supports_persistent = can_use_persistent_tma and (arch == None or int(arch[2:-1]) >= 9)
    if constraints.get("is_persistent", None) is not None:
        is_persistent = constraints["is_persistent"]
    else:
        is_persistent = supports_persistent and (tiles_per_sm >= 10.0 or lhs_dtype.itemsize <= 1)
    # block k
    if constraints.get("block_k", None) is not None:
        block_k = constraints["block_k"]
    else:
        block_k = opt_flags_nvidia.compute_block_k(k, is_persistent, lhs_dtype, rhs_dtype, microscaling_ctx)
    # split_k
    if constraints.get("split_k", None) is not None:
        split_k = constraints["split_k"]
    elif is_persistent or enforce_bitwise_invariance:
        split_k = 1
    else:
        estimated_actual_grid_size = opt_flags_nvidia.compute_grid_size(None, m, n, block_m, block_n)
        split_k = opt_flags_nvidia.compute_split_k(block_k, k, estimated_actual_grid_size)
    if split_k > 1:
        # With split_k, results are written in f32. Use that for the following computations.
        out_dtype = torch.float32
    compute_num_stages_args = (
        precision_config,
        microscaling_ctx,
        is_persistent,
        block_m,
        block_n,
        block_k,
        out_dtype,
        lhs_dtype,
        rhs_dtype,
    )
    if constraints.get("epilogue_subtile", None) is not None:
        epilogue_subtile = constraints["epilogue_subtile"]
    else:
        n1 = opt_flags_nvidia.compute_num_stages(*compute_num_stages_args, False)
        n2 = opt_flags_nvidia.compute_num_stages(*compute_num_stages_args, True)
        epilogue_subtile = n2 > n1 # enable epilogue_subtile if it increases the number of stages
    # num_stages
    num_stages = opt_flags_nvidia.compute_num_stages(*compute_num_stages_args, epilogue_subtile)
    # fused scatter scratchpad
    if constraints.get("fused_scatter", None) is not None:
        fused_scatter = constraints["fused_scatter"]
    else:
        fused_scatter = can_use_fused_scatter and split_k == 1
    # num_warps
    num_warps = opt_flags_nvidia.compute_num_warps(block_m, block_n)
    ret = OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        w_cache_modifier=None,
        split_k=split_k,
        fused_scatter=fused_scatter,
        is_persistent=is_persistent,
        epilogue_subtile=epilogue_subtile,
        arch=arch,
        target_kernel_kwargs=dict(),
    )
    # check constraints
    assert all(getattr(ret, ck) == cv for ck, cv in constraints.items() if cv is not None), f"{ret} != {constraints}"
    return ret

# --------------
# User Interface
# --------------

_opt_flags_constraints: dict = dict()
_opt_flags: OptFlags | None = None

def update_opt_flags_constraints(constraints: dict[str, int]):
    global _opt_flags_constraints
    _opt_flags_constraints.update(constraints)

def reset_opt_flags_constraints():
    global _opt_flags_constraints
    _opt_flags_constraints = None

def set_opt_flags(opt_flags: OptFlags):
    global _opt_flags
    assert not _opt_flags_constraints, "setting constraints is incompatible with manual flags override"
    assert not _opt_flags, "opt_flags already set; please reset to None first"
    _opt_flags = opt_flags


def make_opt_flags(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
):
    microscaling_ctx = precision_config.mx_ctx
    enforce_bitwise_invariance = precision_config.enforce_bitwise_invariance
    if _opt_flags is not None:
        assert not _opt_flags_constraints
        return _opt_flags
    args = [out_dtype, lhs_dtype, rhs_dtype, precision_config, microscaling_ctx, m, n, k,
            routing_data, can_use_persistent_tma, can_use_fused_scatter,
            enforce_bitwise_invariance, _opt_flags_constraints]
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "hip":
        return make_default_opt_flags_amd(*args)
    if backend == "cuda":
        return make_default_opt_flags_nvidia(*args)
    assert False
