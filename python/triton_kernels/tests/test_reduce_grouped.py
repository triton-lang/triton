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
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float16)
    x_tri = x.clone()
    x_ref = x.clone()
    base = torch.arange(0, num_tokens * k, device=device, dtype=torch.int32).view(num_tokens, k)
    # optionally shuffle within each group
    perm = torch.rand((num_tokens, k), device=device).argsort(dim=1)
    indx = base.gather(1, perm)
    mask = (torch.rand((num_tokens, k), device=device) < 0.33)
    indx[mask] = -1
    x_tri, ow_tri = reduce_grouped(x_tri, indx=indx, inplace=True)
    x_ref, ow_ref = reduce_grouped_torch(x_ref, indx=indx)
    assert torch.allclose(x_tri.float(), x_ref.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(ow_tri, ow_ref)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, k, n_cols = 16384, 4, 4096
    dtype, iters = torch.float16, 200
    x = torch.randn((num_tokens * k, n_cols), device=device, dtype=torch.float32).to(dtype)
    indx = torch.randint(0, num_tokens * k, (num_tokens, k), device=device, dtype=torch.int32)
    mask = (torch.rand((num_tokens, k), device=device) < 0.2)
    indx[mask] = -1
    ms = do_bench(lambda: reduce_grouped(x, indx=indx, inplace=True), rep=iters)
    elem = x.element_size()
    valid = (indx.view(-1) != -1)
    reads_rows = int(valid.sum().item())
    writes_rows = int(((indx != -1).any(dim=1)).sum().item())
    index_bytes = num_tokens * k * 4
    bytes_total = reads_rows * n_cols * elem + writes_rows * n_cols * elem + index_bytes
    gbps = (bytes_total) / ms / 1e6
    print(
        f"reduce_grouped: tokens={num_tokens}, N={n_cols}, k={k}, dtype={str(dtype).split('.')[-1]} -> {gbps:.2f} GB/s")
