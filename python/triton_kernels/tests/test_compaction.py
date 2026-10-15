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


@pytest.mark.parametrize("k", [1, 3, 7, 12, 32, 33, 63, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("transpose_mask", [False, True])
def test_compaction_arbitrary_width(k, dtype, index_dtype, transpose_mask, device):
    n_rows, n_experts, sentinel = 7, 128, -7
    indices = [[(17 * col + 19 * row) % n_experts for col in range(k)] for row in range(n_rows)]
    values = [[row + (col - k // 2) * 0.5 for col in range(k)] for row in range(n_rows)]
    masks = []
    expected_values, expected_indices = [], []
    for row in range(n_rows):
        # Include empty/full rows, alternating elements, and either endpoint.
        keep = [
            row == 1 or (row == 2 and col % 2 == 0) or (row == 3 and col == k - 1) or (row == 4 and col == 0)
            or (row == 5 and col % 3 == 1) or (row == 6 and col >= k // 2) for col in range(k)
        ]
        words = [0] * (n_experts // 32)
        selected_values, selected_indices = [], []
        for value, index, active in zip(values[row], indices[row], keep):
            if active:
                words[index // 32] |= 1 << (index % 32)
                selected_values.append(value)
                selected_indices.append(index)
        masks.append(words)
        padding = [sentinel] * (k - len(selected_indices))
        expected_values.append(selected_values + padding)
        expected_indices.append(selected_indices + padding)

    yv = torch.tensor(values, dtype=dtype, device=device)
    yi = torch.tensor(indices, dtype=index_dtype, device=device)
    bitmask = torch.tensor(masks, dtype=torch.uint32, device=device)
    if transpose_mask:
        bitmask = bitmask.T.contiguous().T
    actual_values, actual_indices = compaction(yv, yi, bitmask, sentinel=sentinel)
    torch.testing.assert_close(actual_values, torch.tensor(expected_values, dtype=dtype, device=device), rtol=0, atol=0)
    torch.testing.assert_close(actual_indices, torch.tensor(expected_indices, dtype=index_dtype, device=device), rtol=0,
                               atol=0)
