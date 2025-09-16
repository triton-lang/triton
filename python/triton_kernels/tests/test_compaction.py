import pytest
import torch
from triton_kernels.compaction import compaction, compaction_torch


@pytest.mark.parametrize("n_tokens, n_cols, k, p", [
    (8192, 64, 4, 0.5),
    (8192, 64, 4, 1.0),
    (131, 128, 16, 0.6),
    (496, 128, 16, 0.),
])
def test_compaction(n_tokens, n_cols, k, p, device):
    yi = torch.rand((n_tokens, n_cols), device=device).argsort(dim=-1)
    yi = yi[:, :k].to(torch.int32)
    yv = torch.randn((n_tokens, k), dtype=torch.bfloat16, device=device)
    # "drop" indices from yi with probability `p`
    mask = torch.zeros((n_tokens, n_cols), dtype=torch.int32, device=device)
    keep = (torch.rand(yi.shape, device=device) < p)
    if keep.any():
        rows = torch.arange(yi.size(0), device=device).unsqueeze(1).expand_as(yi)
        mask[rows[keep], yi[keep]] = 1
    chunks = mask.view(*mask.shape[:-1], -1, 32)
    weights = (1 << torch.arange(32, dtype=torch.int32, device=device))
    bitmask = (chunks.int() * weights).sum(dim=-1)
    yv_ref, yi_ref = compaction_torch(yv, yi, bitmask)
    yv_tri, yi_tri = compaction(yv, yi, bitmask)
    assert torch.all(yi_ref == yi_tri)
    assert torch.all(yv_ref == yv_tri)
