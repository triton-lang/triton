import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip, to_numpy

if not is_hip():
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize("block_k", [1, 2, 3, 4])
def test_int8_dot_small_block_k(block_k, device):
    @triton.jit
    def kernel(a, b, c, K, BLOCK_K: tl.constexpr, PAD_K: tl.constexpr):
        offs_m = tl.arange(0, 32)
        offs_n = tl.arange(0, 32)
        offs_k = tl.arange(0, PAD_K)
        a_ptrs = a + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b + offs_k[:, None] * 32 + offs_n[None, :]
        k_mask = offs_k < BLOCK_K
        acc = tl.zeros((32, 32), dtype=tl.int32)
        for _ in range(0, K, BLOCK_K):
            acc = tl.dot(tl.load(a_ptrs, mask=k_mask[None, :], other=0),
                         tl.load(b_ptrs, mask=k_mask[:, None], other=0), acc, out_dtype=tl.int32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * 32
        tl.store(c + offs_m[:, None] * 32 + offs_n[None, :], acc)

    # tl.arange requires a power-of-two range; pad so BLOCK_K=3 stays a logical 3-wide K tile.
    pad_k = triton.next_power_of_2(block_k)
    K = 2052
    rng = np.random.default_rng(0)
    a = rng.integers(120, 128, size=(32, K), dtype=np.int8)
    b = rng.integers(120, 128, size=(K, 32), dtype=np.int8)
    expected = a.astype(np.int64) @ b.astype(np.int64)
    assert np.abs(expected).max() > 2**24

    a = torch.from_numpy(a).to(device)
    b = torch.from_numpy(b).to(device)
    actual = torch.empty((32, 32), dtype=torch.int32, device=device)
    kernel[(1, )](a, b, actual, K, BLOCK_K=block_k, PAD_K=pad_k)
    np.testing.assert_array_equal(to_numpy(actual), expected)
