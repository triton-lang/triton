import pytest
import torch
from triton.testing import do_bench

from triton_kernels.reduce_grouped import reduce_grouped, reduce_grouped_torch


def make_rand_indx(num_tokens: int, k: int, p_invalid: float = 0.25, device: str = "cuda") -> torch.Tensor:
    base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
    perm = torch.rand((num_tokens, k), device=device).argsort(dim=1)
    indx = base.gather(1, perm)
    mask = torch.rand((num_tokens, k), device=device) < p_invalid
    indx[mask] = -1
    return indx


@pytest.mark.parametrize("num_tokens, k, n_cols", [
    (256, 1, 128),
    (256, 2, 256),
    (127, 3, 511),
    (1024, 4, 1024),
])
def test_op(num_tokens, k, n_cols):
    torch.manual_seed(0)
    device = "cuda"
    x_tri = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float16)
    x_ref = x_tri.clone()
    indx = make_rand_indx(num_tokens, k, p_invalid=0.33, device=device)
    x_tri, out_indx_tri = reduce_grouped(x_tri, indx=indx, inplace=True, sanitize=True)
    x_ref, out_indx_ref = reduce_grouped_torch(x_ref, indx=indx)
    assert torch.allclose(x_tri.float(), x_ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(out_indx_tri, out_indx_ref)


def bench_op(num_tokens: int = 16384, k: int = 4, n_cols: int = 4096, dtype: torch.dtype = torch.float16,
             iters: int = 200, p_invalid: float = 0.33):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float32).to(dtype)
    indx = make_rand_indx(num_tokens, k, p_invalid=p_invalid, device=device)
    ms = do_bench(lambda: reduce_grouped(x, indx=indx, inplace=True), rep=iters)
    elem = x.element_size()
    valid = (indx.view(-1) != -1)
    reads_rows = int(valid.sum().item())
    writes_rows = int(((indx != -1).any(dim=1)).sum().item())
    index_bytes = num_tokens * k * 4
    bytes_total = reads_rows * n_cols * elem + writes_rows * n_cols * elem + index_bytes
    gbps = (bytes_total) / ms / 1e6
    print(f"tokens={num_tokens}, N={n_cols}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s")


if __name__ == "__main__":
    bench_op()
