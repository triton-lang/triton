import pytest
import torch
from triton.testing import do_bench
from triton_kernels.reduce import reduce, reduce_torch


def init_mask(mask_mode, B, M, N, device):
    if mask_mode == "none":
        return None
    if mask_mode == "full":
        mask = (torch.rand((B, M, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_b":
        mask = (torch.rand((1, M, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_m":
        mask = (torch.rand((B, 1, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_n":
        mask = (torch.rand((B, M, 1), device=device) > 0.3).to(torch.int8)
    return mask.expand(B, M, N)


@pytest.mark.parametrize("B,M,N", [
    (907, 809, 1001),
    (907, 1024, 1024),
    (1024, 907, 1024),
    (1024, 1024, 907),
    (1024, 1024, 1024),
    (4, 4, 4),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mask_mode", [
    "none",  # no mask
    "full",  # full-sized mask [B,M,N]
    "broadcast_b",  # broadcast over B: [1,M,N]
    "broadcast_m",  # broadcast over M: [B,1,N]
    "broadcast_n",  # broadcast over N: [B,M,1]
])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_op(B, M, N, dtype, dim, mask_mode):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((B, M, N), device=device, dtype=torch.float32).to(dtype)
    mask = init_mask(mask_mode, B, M, N, device)
    y_tri = reduce(x, dim=dim, mask=mask).float()
    y_ref = reduce_torch(x, dim=dim, mask=mask).float()
    assert torch.allclose(y_tri, y_ref, atol=1e-3, rtol=1e-3)


def bench_reduce(B: int = 4, M: int = 4096, N: int = 4096, *, dim: int = 0, dtype: torch.dtype = torch.float16,
                 iters: int = 200, mask_mode: str = "none"):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((B, M, N), device=device, dtype=torch.float32).to(dtype)
    mask = init_mask(mask_mode, B, M, N, device)
    ms = do_bench(lambda: reduce(x, dim=dim, mask=mask), rep=iters)
    nnz = x.numel() if mask is None else (mask.expand(B, M, N) != 0).sum()
    read_bytes = nnz * x.element_size()
    out_elems = (M * N) if dim == 0 else ((B * N) if dim == 1 else (B * M))
    write_bytes = out_elems * x.element_size()
    mask_bytes = 0 if mask is None else (mask.numel() * mask.element_size())
    bytes_total = read_bytes + write_bytes + mask_bytes
    gbps = (bytes_total) / ms / 1e6
    print(
        f"reduce: B={B}, M={M}, N={N}, dim={dim}, dtype={str(dtype).split('.')[-1]}, mask={mask_mode} -> {gbps:.2f} GB/s"
    )


bench_reduce(B=4, M=8192, N=8192, dim=0, dtype=torch.float16, mask_mode="none")
bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_n")
bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_m")
bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_b")
