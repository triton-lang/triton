import triton
import triton.language as tl

# -----------------------------------------------------------------------------
#                                  Utilities
# -----------------------------------------------------------------------------


@triton.constexpr_function
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    tl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


@triton.jit
def compute_pids(block_id, grid_m, grid_n, num_blocks, XCD_SWIZZLE: tl.constexpr, GROUP_M: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    pid_zmnk = block_id
    if XCD_SWIZZLE != 1:
        pid_zmnk = xcd_swizzle(pid_zmnk, num_blocks, XCD_SWIZZLE)
    pid_z = pid_zmnk // (grid_m * grid_n * SPLIT_K)
    pid_mnk = pid_zmnk % (grid_m * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, grid_m, grid_n, GROUP_M)
    return pid_z, pid_m, pid_n, pid_k


@triton.jit
def compute_offsets(
    pid_z,
    pid_m,
    pid_k,
    XBlockSchedule,
    XSliceOffs,
    XBlockOffs,
    X_SLICE_SIZE_DIVISIBILITY: tl.constexpr,
    WBlockSchedule,
    WSliceOffs,
    W_SLICE_SIZE_DIVISIBILITY: tl.constexpr,
    RAGGED_DIMENSION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_X: tl.constexpr,
    PACKED_BLOCK_K_W: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    if RAGGED_DIMENSION == "K":
        # pid_z indicates slice ID: experts are laid sequentially along the K dimension
        # (i.e., we have columns for expert 0, and then expert 1, and then so on).
        # pid_k is meaningless (always zero).
        tl.static_assert(
            X_SLICE_SIZE_DIVISIBILITY is not None or W_SLICE_SIZE_DIVISIBILITY is not None,
            "At least one input must be padded!",
        )
        tl.static_assert(SPLIT_K == 1, "Not supported yet")
        off_x_k = tl.load(XSliceOffs + pid_z)
        off_w_k = tl.load(WSliceOffs + pid_z)
        if PACKED_BLOCK_K_W >= BLOCK_K_X:
            off_w_k = off_w_k * (PACKED_BLOCK_K_W // BLOCK_K_X)
        else:
            off_w_k = off_w_k // (BLOCK_K_X // PACKED_BLOCK_K_W)
        off_x_m = BLOCK_M * pid_m
        off_w_z, off_x_z, off_x_slice, off_x_slice_tile = 0, 0, 0, 0
        off_y_z = pid_z
    elif RAGGED_DIMENSION == "M":
        off_x_k = pid_k * BLOCK_K_X
        off_w_k = pid_k * PACKED_BLOCK_K_W
        block_schedule = tl.load(XBlockSchedule + pid_m)
        off_w_z = block_schedule & 0x0000FFFF
        block_id = block_schedule >> 16
        off_x_slice = tl.load(XSliceOffs + off_w_z)
        off_x_slice_tile = tl.load(XBlockOffs + off_w_z)
        off_x_z, off_y_z = 0, 0
        off_x_m = BLOCK_M * block_id
    else:
        tl.static_assert(RAGGED_DIMENSION is None)
        off_x_k = pid_k * BLOCK_K_X
        off_w_k = pid_k * PACKED_BLOCK_K_W
        off_w_z, off_x_z, off_y_z, off_x_slice, off_x_slice_tile = pid_z, pid_z, pid_z, 0, 0
        off_x_m = BLOCK_M * pid_m
    return (
        off_w_z,
        off_x_z,
        off_y_z,
        off_x_slice,  # offset for the current slice vs 0
        off_x_slice_tile,  # block offset for the current slice vs 0
        off_x_m,  # offset for the current block vs slice start
        off_x_k,
        off_w_k,
    )


def make_matmul_repr(base_name, order):

    def matmul_repr(specialization):
        signature = specialization.signature
        constants = specialization.constants
        reorder = lambda L: [L[i] for i in order]
        layout = lambda stride: "N" if stride in constants else "T"

        def convert_dtype(dtype):
            if "tensordesc" in dtype:
                ret = convert_dtype(dtype.split("<")[1].split("[")[0])
                return ret
            elif "u8" in dtype:
                return "mxfp4"
            elif dtype[0] == "*":
                return dtype[1:]
            else:
                return dtype

        dtypes = "x".join([convert_dtype(f"{signature[i]}") for i in reorder(["Y", "X", "W"])])
        layouts = "".join([f"{layout(i)}" for i in reorder(["stride_y_n", "stride_x_k", "stride_w_n"])])
        blocks = "x".join([f"{constants[i]}" for i in ["BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"]])
        suffix = "_acc" if "OutAcc" in signature and "OutAcc" not in constants else ""
        # mode = []
        # if "GatherIndx" not in constants:
        #     mode += ['g']
        # if "ScatterSrcIndx" not in constants:
        #     mode += ['s']
        # suffix = "" if not mode else "_o" + (''.join(mode))
        # if base_name.startswith("_p"):
        #     suffix += "_ptma"
        return f"{base_name}{suffix}_{layouts}_{dtypes}_{blocks}"

    return matmul_repr


def matmul_launch_metadata(grid, kernel, args):
    from ..proton_opts import launch_metadata_allow_sync

    ret = dict()
    M, N, K = args["M"], args["N"], args["K"]
    Y, X, W = args["YPtr"], args["XPtr"], args["WPtr"]
    expected_slice_sizes = args.get("X_EXPECTED_SLICE_SIZE")
    slice_sizes = args["XSliceSizes"]
    batch_size = args.get("batch_size", 1)
    n_rows = "unknown"
    if expected_slice_sizes is not None:
        n_rows = f"{expected_slice_sizes}*"
    elif slice_sizes is not None and launch_metadata_allow_sync():
        n_rows = int(slice_sizes.float().mean())

    n_tokens = None
    if slice_sizes is not None:
        if launch_metadata_allow_sync():
            n_tokens = int(slice_sizes.sum())
        else:
            n_tokens = slice_sizes.sum()  # n_tokens can stay in gpu

    K_repr = K
    if args["RAGGED_DIMENSION"] == "K":
        K = None if n_tokens is None else n_tokens
        K_repr = K if launch_metadata_allow_sync(
        ) else None  # make sure K_repr is string compatible as K can be on a GPU tensor

    repr = lambda s, x: f"{s} = {x}" if x is not None else f"E_{len(slice_sizes)}({s}) = {n_rows}"
    nbits = X.dtype.itemsize * 8
    batch_repr = ""
    if batch_size > 1:
        batch_repr = repr("B", args["batch_size"]) + ", "
    ret["name"] = (
        f"{kernel.name} [{batch_repr}{repr('M', M)}, {repr('N', N)}, {repr('K', K_repr)}] stg{kernel.num_stages}")
    ep_subtile = args["EPILOGUE_SUBTILE"]
    if ep_subtile is not None and ep_subtile > 1:
        ret["name"] += f" ep/{ep_subtile}"

    if slice_sizes is not None and n_tokens is None:
        return ret  # Don't fill metadata because we can't compute them properly.

    fM = M if M is not None else n_tokens
    Z = 1 if args["RAGGED_DIMENSION"] == "K" else batch_size
    ret[f"flops{nbits}"] = 2.0 * fM * N * K * Z

    # sindx = args.get("WriteBackIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    n_w_bytes = W.numel() * W.element_size()
    if slice_sizes is not None:
        assert n_tokens is not None
        n_read_rows = n_tokens

        if args["RAGGED_DIMENSION"] == "K":
            n_x_bytes = n_read_rows * X.shape[-2] * X.element_size()
            # Here, we're computing dW = X.T@dY, so "W" is actually dY and "Y" is actually dW.
            n_y_bytes = Y.numel() * Y.element_size() * (2 if args["OutAcc"] is not None else 1)
            n_w_bytes = n_read_rows * W.shape[-1] * W.element_size()
        else:
            n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
            n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
            n_w_bytes = (W.numel() * W.element_size() // slice_sizes.numel()) * (slice_sizes > 0).sum()

    ret["bytes"] = n_x_bytes + n_y_bytes + n_w_bytes
    return ret


@triton.jit
def threadfence_system():
    tl.inline_asm_elementwise("mov.u32 $0, 0x0; fence.sc.sys;", args=(), dtype=(tl.int32, ), is_pure=False, pack=1,
                              constraints="=r")
