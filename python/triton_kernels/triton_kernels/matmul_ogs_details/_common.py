import triton
import triton.language as tl

# -----------------------------------------------------------------------------
#                                  Utilities
# -----------------------------------------------------------------------------


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
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


def make_matmul_repr(base_name, order):

    def matmul_repr(specialization):
        signature = specialization.signature
        constants = specialization.constants
        reorder = lambda L: [L[i] for i in order]
        layout = lambda stride: "N" if stride in constants else "T"
        convert_dtype = lambda dtype: "mxfp4" if "u8" in dtype else dtype
        dtypes = "x".join([convert_dtype(f"{signature[i][1:]}") for i in reorder(["Y", "X", "W"])])
        layouts = "".join([f"{layout(i)}" for i in reorder(["stride_y_n", "stride_x_k", "stride_w_n"])])
        blocks = "x".join([f"{constants[i]}" for i in ["BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"]])
        mode = []
        if "GatherIndx" not in constants:
            mode += ['g']
        if "ScatterSrcIndx" not in constants:
            mode += ['s']
        suffix = "" if not mode else "_o" + (''.join(mode))
        if base_name.startswith("_p"):
            suffix += "_ptma"
        return f"_matmul{suffix}_{layouts}_{dtypes}_{blocks}"

    return matmul_repr


def matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = args["M"], args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    if hist is not None:
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()

        # If annotation is given, use that to generate name for profiling.
        tokens_per_expt = args.get("TOKENS_PER_EXPT_FOR_ANNOTATION")
        n_rows = f"{tokens_per_expt}*" if tokens_per_expt is not None else int(hist.float().mean())
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()
    repr = lambda s, x: f"{s} = {x}" if x is not None else f"E_{len(hist)}({s}) = {n_rows}"
    nbits = X.dtype.itemsize * 8
    batch_repr = ""
    if "batch_size" in args and args["batch_size"] > 1:
        batch_repr = repr("B", args["batch_size"]) + ", "
    ret["name"] = f"{kernel.name}[{batch_repr}{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    fM = M if M is not None else n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK
    gindx = args.get("GatherIndx", None)
    sindx = args.get("WriteBackIndx", None)
    sskipped = 0. if sindx is None else (sindx == -1).sum() / sindx.shape[0]
    gskipped = 0. if gindx is None else (gindx == -1).sum() / gindx.shape[0]
    ret["bytes"] = int((1 - sskipped) * Y.numel() * Y.element_size() + (1 - gskipped) * X.numel() * X.element_size() +
                       n_w_bytes)
    return ret
