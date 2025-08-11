import pytest
import torch
from triton.testing import do_bench
from triton_kernels.scatter import scatter, scatter_torch


def make_rand_row_indx(num_tokens: int, k: int, p_invalid: float = 0.25, device: str = "cuda") -> torch.Tensor:
    row_indx = torch.randint(0, num_tokens * k, (num_tokens, ), device=device, dtype=torch.int32)
    if p_invalid > 0:
        mask = torch.rand((num_tokens, ), device=device) < p_invalid
        row_indx[mask] = -1
    return row_indx


@pytest.mark.parametrize("num_tokens, k, n_cols", [
    (256, 1, 128),
    (256, 2, 256),
    (127, 3, 511),
])
def test_op(num_tokens, k, n_cols):
    torch.manual_seed(1)
    device = "cuda"
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float16)
    row_indx = make_rand_row_indx(num_tokens, k, p_invalid=0.25, device=device)
    out_tri = scatter(x, indx=row_indx)
    out_torch = scatter_torch(x, row_indx=row_indx)
    assert torch.allclose(out_tri.float(), out_torch.float(), atol=1e-3, rtol=1e-3)


def bench_op(num_tokens: int = 16384, k: int = 4, n_cols: int = 4096, dtype: torch.dtype = torch.float16,
             iters: int = 200, p_invalid: float = 0.2):
    torch.manual_seed(1)
    device = "cuda"
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float32).to(dtype)
    row_indx = make_rand_row_indx(num_tokens, k, p_invalid=p_invalid, device=device)
    ms = do_bench(lambda: scatter(x, indx=row_indx), rep=iters)
    elem = x.element_size()
    reads_rows = int((row_indx != -1).sum().item())
    index_bytes = num_tokens * 4
    bytes_total = reads_rows * n_cols * elem + num_tokens * n_cols * elem + index_bytes
    gbps = (bytes_total) / ms / 1e6
    print(f"scatter: tokens={num_tokens}, N={n_cols}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s")


if __name__ == "__main__":
    bench_op()
