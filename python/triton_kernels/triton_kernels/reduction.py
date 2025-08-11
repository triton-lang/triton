import torch
import triton
import triton.language as tl
from triton.testing import do_bench
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import routing


@triton.jit
def _reduce_grouped(X, stride_xm, stride_xn,  #
                    InIndx, N, OutIndx,  #
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    # persistent along N: single program on N, iterate tiles of size BLOCK_N
    start = pid_t * K
    # load indices into a tuple
    indxs = ()
    for i in tl.static_range(0, K):
        indxs = indxs + (tl.load(InIndx + start + i), )
    # determine first valid topk row
    fi = indxs[(K - 1)]
    for i in tl.static_range(K - 2, -1, -1):
        fi = tl.where(indxs[i] != -1, indxs[i], fi)
    # record overwritten row index (may be -1 if none)
    tl.store(OutIndx + pid_t, fi)
    ColPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # accumulate contributions for this tile
        for i in tl.static_range(0, K):
            is_valid = indxs[i] != -1
            row_ptr = ColPtrs + indxs[i] * stride_xm
            vals = tl.load(row_ptr, mask=n_mask & is_valid, other=0.0)
            acc += vals.to(tl.float32)
        # write-back for this tile
        out_ptr = ColPtrs + fi * stride_xm
        tl.store(out_ptr, acc, mask=n_mask & (fi != -1))
        ColPtrs += BLOCK_N * stride_xn


def reduce_grouped(x: torch.Tensor, indx: torch.Tensor, inplace: bool):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]
    - inplace: bool

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]`. If `inplace=True`, overwrites the row corresponding
    to the first valid (non-negative) index with the per-group sum. Accumulation is
    performed in float32 for numerical stability, and the result is written back
    in the dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group and the values in the corresponding row is undefined.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    assert inplace, "only inplace=True is supported for now"
    assert x.shape[0] == indx.numel()
    num_groups = indx.shape[0]
    out_indx = torch.empty((num_groups, ), dtype=torch.int32, device=x.device)
    BLOCK_N = 512
    _reduce_grouped[(num_groups, )](
        x, x.stride(0), x.stride(1),  #
        indx, x.shape[1], out_indx,  #
        BLOCK_N=BLOCK_N, K=indx.shape[1],  #
        num_warps=1,  #
    )
    return x, out_indx


def reduce_grouped_torch(x: torch.Tensor, scatter_indx):
    assert scatter_indx.ndim == 2
    num_tokens, k = scatter_indx.shape
    # Handle possible -1 entries in scatter_indx.src_indx by masking
    src = scatter_indx.src_indx.to(torch.long)
    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    valid = src != -1
    # Sum contributions per token across valid expert rows (in float32)
    token_id = positions // k
    valid_pos = torch.nonzero(valid, as_tuple=False).squeeze(1)
    contrib = x.index_select(0, src.index_select(0, valid_pos)).to(torch.float32)
    sums = torch.zeros((num_tokens, x.shape[-1]), dtype=torch.float32, device=x.device)
    sums.index_add_(0, token_id.index_select(0, valid_pos), contrib)
    # Find first valid gate position per token and map back to topk row index
    big = torch.full_like(src, src.numel(), dtype=torch.long)
    masked_pos = torch.where(valid, positions, big)
    first_pos_flat = masked_pos.view(num_tokens, k).min(dim=1).values
    valid_token_mask = first_pos_flat != big[0]
    overwritten = -torch.ones((num_tokens, ), dtype=torch.int32, device=x.device)
    if valid_token_mask.any():
        idx_keep = torch.nonzero(valid_token_mask, as_tuple=False).squeeze(1)
        first_rows_topk = src.index_select(0, first_pos_flat[valid_token_mask])
        x.index_copy_(0, first_rows_topk, sums.index_select(0, idx_keep).to(x.dtype))
        overwritten.index_copy_(0, idx_keep, first_rows_topk)
    return x, overwritten


@triton.jit
def _scatter_kernel(
    X,
    stride_xm,
    stride_xn,
    RowIndx,
    OUT,
    stride_om,
    stride_on,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    # Load selected row index for this token
    fi = tl.load(RowIndx + pid_t)
    InColPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OutColPtrs = OUT + pid_t * stride_om + tl.arange(0, BLOCK_N) * stride_on
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=3):
        n_mask = tl.arange(0, BLOCK_N) < (N - n_curr)
        eff_fi = tl.where(fi == -1, 0, fi)
        in_ptr = InColPtrs + eff_fi * stride_xm
        vals = tl.load(in_ptr, mask=n_mask & (fi != -1), other=0.0)
        tl.store(OutColPtrs, vals, mask=n_mask)
        InColPtrs += BLOCK_N * stride_xn
        OutColPtrs += BLOCK_N * stride_on


def scatter(x: torch.Tensor, indx: torch.Tensor):
    """
    Row-wise scatter (index-based row selection).

    This copies rows from `x` into a new tensor according to `indx`, where each
    entry of `indx` is a row index into `x`. An index of -1 produces a zero row.

    Arguments
    - x: Tensor of shape [num_rows, N]. Any floating dtype is supported.
    - indx: Tensor[int32] of shape [M]. Each value is either -1 or an integer
      in [0, num_rows). Values outside this range are undefined behavior.

    Returns
    - out: Tensor of shape [M, N] where out[i, :] = x[indx[i], :] if indx[i] != -1,
      otherwise zeros

    Notes
    - This function does not reorder or reduce data beyond selecting rows.
    - The underlying kernel tiles and iterates across columns persistently to
      minimize launch overhead and maximize bandwidth.
    """
    assert indx.ndim == 1
    num_tokens = indx.shape[0]
    n_cols = x.shape[-1]
    out = torch.zeros((num_tokens, n_cols), dtype=x.dtype, device=x.device)
    BLOCK_N = 1024
    _scatter_kernel[(num_tokens, )](
        x,
        x.stride(0),
        x.stride(1),
        indx,
        out,
        out.stride(0),
        out.stride(1),
        n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def scatter_torch(x: torch.Tensor, row_indx: torch.Tensor):
    """
    Torch reference for scatter using per-token row indices.
    - x: [(num_tokens*K), N]
    - row_indx: [num_tokens], row indices in `x` (or -1)
    """
    num_tokens = row_indx.shape[0]
    out = torch.zeros((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
    valid = row_indx != -1
    if valid.any():
        rows = x.index_select(0, row_indx[valid].to(torch.long))
        out.index_copy_(0, torch.nonzero(valid, as_tuple=False).squeeze(1), rows)
    return out


def bench_reduce_grouped(num_tokens: int, n_features: int, k: int, n_experts_tot: int = 8, simulated_ep: int = 1,
                         dtype=torch.float8_e4m3fn, iters: int = 200):
    assert torch.cuda.is_available()
    device = "cuda"
    # Build routing on dummy logits
    logits = torch.randn((num_tokens, n_experts_tot), device=device, dtype=torch.float32)
    routing_data, _, scatter_indx = routing(logits, k, simulated_ep=simulated_ep)
    reduce_indx = scatter_indx.src_indx.view(-1, routing_data.n_expts_act)
    k = int(routing_data.n_expts_act)
    x = torch.randn((num_tokens * k, n_features), device=device, dtype=torch.float32).to(dtype)
    # Benchmark full wrapper (includes minimal Python + launch overhead)
    ms = do_bench(lambda: reduce_grouped(x, indx=reduce_indx, inplace=True), rep=iters)
    elem = x.element_size()
    # Account for invalid tokens: only valid rows are read, writes only when there is at least one valid row
    valid = (scatter_indx.src_indx != -1)
    valid2d = valid.view(num_tokens, k)
    reads_rows = int(valid.sum().item())
    writes_rows = int((valid2d.any(dim=1)).sum().item())
    index_bytes = num_tokens * k * 4
    bytes_total = reads_rows * n_features * elem + writes_rows * n_features * elem + index_bytes
    gbps = (bytes_total) / ms / 1e6
    print(
        f"reduce_inplace: tokens={num_tokens}, N={n_features}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s"
    )


def bench_scatter(num_tokens: int, n_features: int, k: int, n_experts_tot: int = 8, simulated_ep: int = 1,
                  dtype=torch.float16, iters: int = 200):
    assert torch.cuda.is_available()
    device = "cuda"
    # Build routing on dummy logits
    logits = torch.randn((num_tokens, n_experts_tot), device=device, dtype=torch.float32)
    routing_data, _, scatter_indx = routing(logits, k, simulated_ep=simulated_ep)
    reduce_indx = scatter_indx.src_indx.view(-1, routing_data.n_expts_act)
    k = int(routing_data.n_expts_act)
    x = torch.randn((num_tokens * k, n_features), device=device, dtype=torch.float32).to(dtype)
    # Precompute per-token selected row indices via reduce_grouped (no timing)
    _, row_indx = reduce_grouped(x, indx=reduce_indx, inplace=True)
    # Benchmark wrapper call
    ms = do_bench(lambda: scatter(x, indx=row_indx), rep=iters)
    s = ms * 1e-3
    elem = x.element_size()
    # Account for invalid tokens: read only if at least one valid row; always write OUT
    valid = (scatter_indx.src_indx != -1).view(num_tokens, k)
    reads_rows = int((valid.any(dim=1)).sum().item())
    index_bytes = num_tokens * k * 4
    bytes_total = reads_rows * n_features * elem + num_tokens * n_features * elem + index_bytes
    gbps = (bytes_total) / s / 1e9
    print(f"scatter: tokens={num_tokens}, N={n_features}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s")


def _example_scatter_matmul():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping example.")
        return

    torch.manual_seed(0)
    device = "cuda"

    # Problem sizes
    num_tokens = 8192
    in_features = 4096
    out_features = 4096
    num_experts_total = 8
    num_experts_active = 4

    # Inputs and per-expert weights
    x = torch.randn((num_tokens, in_features), device=device, dtype=torch.float16)
    w = torch.randn((num_experts_total, in_features, out_features), device=device, dtype=torch.float16)

    # Routing logits -> routing data and indices
    logits = torch.randn((num_tokens, num_experts_total), device=device, dtype=torch.float32)
    routing_data, gather_indx, scatter_indx = routing(logits, num_experts_active, simulated_ep=1)

    assert scatter_indx is not None, "scatter_indx should not be None"

    # Matmul and manual reduction + scatter to match fused path
    y1_tmp = matmul_ogs(x, w, bias=None, routing_data=routing_data, gather_indx=gather_indx)
    y1_tmp, row_indx = reduce_grouped(y1_tmp, indx=scatter_indx.src_indx.view(-1, routing_data.n_expts_act),
                                      inplace=True)
    y1 = scatter(y1_tmp, indx=row_indx)
    y2 = matmul_ogs(x, w, bias=None, routing_data=routing_data, gather_indx=gather_indx, scatter_indx=scatter_indx)

    # Validate equivalence
    if y1.shape != y2.shape:
        print(f"Shape mismatch: {y1.shape} vs {y2.shape}")
    else:
        max_diff = (y1 - y2).abs().max().item() if y1.numel() > 0 else 0.0
        print(f"Shapes: {y1.shape}; max abs diff: {max_diff}")

    # Benchmarks (tune sizes as needed to approach peak BW)
    print("\n-- Benchmarks --")
    bench_tokens = 16384  # ~256K tokens
    bench_N = out_features
    bench_k = num_experts_active
    bench_reduce_grouped(bench_tokens, bench_N, bench_k, n_experts_tot=num_experts_total, simulated_ep=2,
                         dtype=torch.float16, iters=200)
    bench_scatter(bench_tokens, bench_N, bench_k, n_experts_tot=num_experts_total, simulated_ep=2, dtype=torch.float16,
                  iters=200)


if __name__ == "__main__":
    _example_scatter_matmul()
