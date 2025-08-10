import torch
import triton
import triton.language as tl
from triton.testing import do_bench
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import routing


@triton.jit
def _reduce_inplace_kernel(
    X,
    stride_xm,
    stride_xn,
    SRC,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    start = pid_t * K
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # determine first valid topk row (leftmost where SRC != -1) by scanning right-to-left
    fi = tl.load(SRC + start + (K - 1))
    for i in range(K - 2, -1, -1):
        idx = tl.load(SRC + start + i)
        fi = tl.where(idx != -1, idx, fi)

    for i in range(0, K):
        topk_idx = tl.load(SRC + start + i)
        is_valid = topk_idx != -1
        row_ptr = X + topk_idx * stride_xm + offs_n * stride_xn
        vals = tl.load(row_ptr, mask=n_mask & is_valid, other=0.0)
        acc += vals.to(tl.float32)

    out_ptr = X + fi * stride_xm + offs_n * stride_xn
    tl.store(out_ptr, acc, mask=n_mask & (fi != -1))


def reduce_inplace(x: torch.Tensor, routing_data, scatter_indx):
    k = int(routing_data.n_expts_act)
    if k <= 1 or x.numel() == 0:
        return x
    src = scatter_indx.src_indx
    assert src.dim() == 1
    num_tokens = src.numel() // k
    n_cols = x.shape[-1]

    BLOCK_N = 1024
    grid = (num_tokens, triton.cdiv(n_cols, BLOCK_N))
    _reduce_inplace_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        src,
        n_cols,
        K=k,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return x


def reduce_inplace_torch(x: torch.Tensor, routing_data, scatter_indx):
    """
    For each token, overwrite the row corresponding to the first expert with
    the sum over that token's expert rows. Operates in-place on `x`.

    x is expected to be in topk/gather order with shape [num_tokens * k, out_features].
    """
    k = routing_data.n_expts_act
    if k <= 1 or x.numel() == 0:
        return x
    # Handle possible -1 entries in scatter_indx.src_indx by masking
    src = scatter_indx.src_indx.to(torch.long)
    num_tokens = src.numel() // k
    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    valid = src != -1
    if not valid.any():
        return x
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
    if valid_token_mask.any():
        first_rows_topk = src.index_select(0, first_pos_flat[valid_token_mask])
        x.index_copy_(0, first_rows_topk,
                      sums.index_select(0,
                                        torch.nonzero(valid_token_mask, as_tuple=False).squeeze(1)).to(x.dtype))
    return x


@triton.jit
def _scatter_kernel(
    X,
    stride_xm,
    stride_xn,
    SRC,
    OUT,
    stride_om,
    stride_on,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    start = pid_t * K
    # find first valid topk row (leftmost where SRC != -1) by scanning right-to-left
    fi = tl.load(SRC + start + (K - 1))
    for i in range(K - 2, -1, -1):
        idx = tl.load(SRC + start + i)
        fi = tl.where(idx != -1, idx, fi)

    # Avoid invalid pointer when no valid row: map to row 0 but fully masked by (fi != -1)
    eff_fi = tl.where(fi == -1, 0, fi)
    in_ptr = X + eff_fi * stride_xm + offs_n * stride_xn
    vals = tl.load(in_ptr, mask=n_mask & (fi != -1), other=0.0)

    out_ptr = OUT + pid_t * stride_om + offs_n * stride_on
    tl.store(out_ptr, vals, mask=n_mask)


def scatter(x: torch.Tensor, routing_data, scatter_indx):
    k = int(routing_data.n_expts_act)
    if k <= 1 or x.numel() == 0:
        return x
    src = scatter_indx.src_indx
    num_tokens = src.numel() // k
    n_cols = x.shape[-1]
    out = torch.zeros((num_tokens, n_cols), dtype=x.dtype, device=x.device)
    BLOCK_N = 1024
    grid = (num_tokens, triton.cdiv(n_cols, BLOCK_N))
    _scatter_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        src,
        out,
        out.stride(0),
        out.stride(1),
        n_cols,
        K=k,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def scatter_torch(x: torch.Tensor, routing_data, scatter_indx):
    """
    For each token, extract the row corresponding to the first expert into a
    new tensor with shape [num_tokens, out_features].
    """
    k = int(routing_data.n_expts_act)
    if k <= 1 or x.numel() == 0:
        return x
    src = scatter_indx.src_indx.to(torch.long)
    num_tokens = src.numel() // k
    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    valid = src != -1
    big = torch.full_like(src, src.numel(), dtype=torch.long)
    masked_pos = torch.where(valid, positions, big)
    first_pos_flat = masked_pos.view(num_tokens, k).min(dim=1).values
    valid_token_mask = first_pos_flat != big[0]
    # Build output; default zeros for any tokens with no valid entries (should be rare)
    out = torch.zeros((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
    if valid_token_mask.any():
        first_rows_topk = src.index_select(0, first_pos_flat[valid_token_mask])
        out.index_copy_(0,
                        torch.nonzero(valid_token_mask, as_tuple=False).squeeze(1), x.index_select(0, first_rows_topk))
    return out


def bench_reduce_inplace(num_tokens: int, n_features: int, k: int, n_experts_tot: int = 8, simulated_ep: int = 1,
                         dtype=torch.float16, iters: int = 200):
    assert torch.cuda.is_available()
    device = "cuda"
    # Build routing on dummy logits
    logits = torch.randn((num_tokens, n_experts_tot), device=device, dtype=torch.float32)
    routing_data, gather_indx, scatter_indx = routing(logits, k, simulated_ep=simulated_ep)
    k = int(routing_data.n_expts_act)
    x = torch.randn((num_tokens * k, n_features), device=device, dtype=dtype)
    # Benchmark full wrapper (includes minimal Python + launch overhead)
    ms = do_bench(lambda: reduce_inplace(x, routing_data, scatter_indx), rep=iters)
    s = ms * 1e-3
    elem = x.element_size()
    # Account for invalid tokens: only valid rows are read, writes only when there is at least one valid row
    valid = (scatter_indx.src_indx != -1)
    valid2d = valid.view(num_tokens, k)
    reads_rows = int(valid2d.sum().item())
    writes_rows = int((valid2d.any(dim=1)).sum().item())
    index_bytes = num_tokens * k * 4
    bytes_total = reads_rows * n_features * elem + writes_rows * n_features * elem + index_bytes
    gbps = (bytes_total) / s / 1e9
    print(
        f"reduce_inplace: tokens={num_tokens}, N={n_features}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s"
    )


def bench_scatter(num_tokens: int, n_features: int, k: int, n_experts_tot: int = 8, simulated_ep: int = 1,
                  dtype=torch.float16, iters: int = 200):
    assert torch.cuda.is_available()
    device = "cuda"
    # Build routing on dummy logits
    logits = torch.randn((num_tokens, n_experts_tot), device=device, dtype=torch.float32)
    routing_data, gather_indx, scatter_indx = routing(logits, k, simulated_ep=simulated_ep)
    k = int(routing_data.n_expts_act)
    x = torch.randn((num_tokens * k, n_features), device=device, dtype=dtype)
    # Benchmark wrapper call (allocates output each call, matching typical usage)
    ms = do_bench(lambda: scatter(x, routing_data, scatter_indx), rep=iters)
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
    y1_tmp = reduce_inplace(y1_tmp, routing_data=routing_data, scatter_indx=scatter_indx)
    y1 = scatter(y1_tmp, routing_data=routing_data, scatter_indx=scatter_indx)
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
    bench_reduce_inplace(bench_tokens, bench_N, bench_k, n_experts_tot=num_experts_total, simulated_ep=2,
                         dtype=torch.float16, iters=200)
    bench_scatter(bench_tokens, bench_N, bench_k, n_experts_tot=num_experts_total, simulated_ep=2, dtype=torch.float16,
                  iters=200)


if __name__ == "__main__":
    _example_scatter_matmul()
