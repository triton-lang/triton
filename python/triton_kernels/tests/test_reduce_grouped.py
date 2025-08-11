import pytest
import torch
from triton.testing import do_bench

from triton_kernels.reduce_grouped import reduce_grouped, reduce_grouped_torch


@pytest.mark.parametrize("num_tokens, k, n_cols", [
    (256, 1, 128),
    (256, 2, 256),
    (127, 3, 511),
    (1024, 4, 1024),
])
def test_op(num_tokens, k, n_cols):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(0)

    # Build random x and random indices with some -1s
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float16)
    # start with sequential rows then randomly drop entries to -1
    base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
    # random mask of invalids (~25%)
    mask = (torch.rand((num_tokens, k), device=device) < 0.25)
    indx = base.clone()
    indx[mask] = -1
    # Ensure at least one -1 is present deterministically
    indx[0, 0] = -1

    # Triton
    x_tri = x.clone()
    x_tri, overwritten_tri = reduce_grouped(x_tri, indx=indx, inplace=True)

    # Torch
    x_torch = x.clone()
    x_torch, overwritten_torch = reduce_grouped_torch(x_torch, indx=indx)

    # Compare
    assert torch.allclose(x_tri.float(), x_torch.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(overwritten_tri, overwritten_torch)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        torch.manual_seed(0)
        device = "cuda"
        # Benchmark parameters (tune as desired)
        num_tokens = 16384
        k = 4
        n_cols = 4096
        dtype = torch.float16
        iters = 200

        # Build inputs
        x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float32).to(dtype)
        base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
        mask = (torch.rand((num_tokens, k), device=device) < 0.2)
        indx = base.clone()
        indx[mask] = -1

        def _call():
            reduce_grouped(x, indx=indx, inplace=True)

        ms = do_bench(_call, rep=iters)
        elem = x.element_size()
        valid = (indx.view(-1) != -1)
        reads_rows = int(valid.sum().item())
        writes_rows = int(((indx != -1).any(dim=1)).sum().item())
        index_bytes = num_tokens * k * 4
        bytes_total = reads_rows * n_cols * elem + writes_rows * n_cols * elem + index_bytes
        gbps = (bytes_total) / ms / 1e6
        print(
            f"reduce_grouped: tokens={num_tokens}, N={n_cols}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s"
        )
