import pytest
import torch
from triton.testing import do_bench

from triton_kernels.scatter import scatter, scatter_torch
from triton_kernels.reduce_grouped import reduce_grouped


@pytest.mark.parametrize("num_tokens, k, n_cols", [
    (256, 1, 128),
    (256, 2, 256),
    (127, 3, 511),
])
def test_op(num_tokens, k, n_cols):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(1)

    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float16)
    base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
    mask = (torch.rand((num_tokens, k), device=device) < 0.25)
    indx = base.clone()
    indx[mask] = -1

    # Derive per-token selected row indices using reduce_grouped (no need for correctness here)
    _, row_indx = reduce_grouped(x.clone(), indx=indx, inplace=True)

    # Triton
    out_tri = scatter(x, indx=row_indx)

    # Torch
    out_torch = scatter_torch(x, row_indx=row_indx)

    assert torch.allclose(out_tri.float(), out_torch.float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        torch.manual_seed(1)
        device = "cuda"
        # Benchmark parameters
        num_tokens = 16384
        k = 4
        n_cols = 4096
        dtype = torch.float16
        iters = 200

        x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float32).to(dtype)
        base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
        mask = (torch.rand((num_tokens, k), device=device) < 0.2)
        indx = base.clone()
        indx[mask] = -1
        # Precompute per-token selected row indices (not timed)
        _, row_indx = reduce_grouped(x.clone(), indx=indx, inplace=True)

        def _call():
            scatter(x, indx=row_indx)

        ms = do_bench(_call, rep=iters)
        elem = x.element_size()
        # Reads only when at least one valid per token; always write one row per token
        valid = (indx != -1)
        reads_rows = int((valid.any(dim=1)).sum().item())
        index_bytes = num_tokens * 4
        bytes_total = reads_rows * n_cols * elem + num_tokens * n_cols * elem + index_bytes
        gbps = (bytes_total) / ms / 1e6
        print(f"scatter: tokens={num_tokens}, N={n_cols}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s")
