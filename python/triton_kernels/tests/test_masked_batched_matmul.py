# isort: off
# fmt: off
"""Masked batched matmul: only the valid leading rows of each expert are computed.

Contract under test (see the matmul_ogs docstring): for expert e, rows
[0, masked_m[e]) are real tokens and rows [masked_m[e], M_capacity) are padding
that lies outside the result contract. The reference is therefore the *dense*
batched matmul on the same buffers, compared on the valid prefix only.
"""
import pytest
import torch

from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.testing import assert_close


def _valid_row_patterns(E, Mcap):
    """(id, valid_counts) covering the occupancy shapes that matter."""
    cases = [
        ("all_full", [Mcap] * E),
        ("all_empty", [0] * E),
        ("one_token_each", [1] * E),
        ("half", [Mcap // 2] * E),
        ("one_empty_expert", [Mcap] * (E - 1) + [0]),
        ("only_first_nonempty", [Mcap] + [0] * (E - 1)),
        ("only_last_nonempty", [0] * (E - 1) + [Mcap]),
    ]
    if Mcap > 128:
        cases += [
            # Around the resolved BLOCK_M (128 for these shapes): the tile that
            # starts exactly at the capacity boundary is the interesting one.
            ("last_tile_partial", [128 + 1] * E),
            ("last_tile_empty", [128] * E),
            ("ragged", [(i * 37 + 1) % Mcap for i in range(E)]),
            ("capacity_minus_one", [Mcap - 1] * E),
        ]
    return cases


@pytest.mark.parametrize("E,Mcap,K,N", [
    (1, 512, 1024, 1024),
    (8, 128, 1024, 1024),
    (8, 256, 1024, 2048),
    (8, 512, 1024, 1024),
    (8, 512, 2048, 2048),
    (16, 512, 1024, 1024),
])
def test_masked_batched_matmul_matches_dense_on_valid_rows(E, Mcap, K, N, device):
    torch.manual_seed(20260918)
    x = torch.randn(E, Mcap, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) * 0.02
    bias = torch.zeros(E, N, device=device, dtype=torch.float32)

    dense = matmul_ogs(x, w, bias)

    for case, valid in _valid_row_patterns(E, Mcap):
        mm = torch.tensor(valid, device=device, dtype=torch.int32)
        out = matmul_ogs(x, w, bias, masked_m=mm)
        assert out.shape == dense.shape, case
        for e in range(E):
            v = valid[e]
            if v == 0:
                continue
            assert_close(out[e, :v], dense[e, :v], maxtol=2e-2, rmstol=2e-2,
                         description=f"{case}: expert {e}", verbose=False)


def test_masked_batched_matmul_full_mask_is_bit_exact(device):
    """masked_m == capacity leaves no skippable tile, so it must not change bits."""
    torch.manual_seed(20260918)
    E, Mcap, K, N = 8, 512, 1024, 1024
    x = torch.randn(E, Mcap, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) * 0.02
    bias = torch.zeros(E, N, device=device, dtype=torch.float32)
    mm = torch.full((E,), Mcap, device=device, dtype=torch.int32)

    dense = matmul_ogs(x, w, bias)
    masked = matmul_ogs(x, w, bias, masked_m=mm)
    assert torch.equal(masked, dense)


def test_masked_batched_matmul_rejects_non_batched_input(device):
    x = torch.randn(512, 1024, device=device, dtype=torch.bfloat16)
    w = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    bias = torch.zeros(1024, device=device, dtype=torch.float32)
    mm = torch.full((1,), 512, device=device, dtype=torch.int32)
    with pytest.raises(AssertionError):
        matmul_ogs(x, w, bias, masked_m=mm)


def test_masked_batched_matmul_rejects_wrong_length(device):
    E, Mcap, K, N = 8, 512, 1024, 1024
    x = torch.randn(E, Mcap, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16)
    bias = torch.zeros(E, N, device=device, dtype=torch.float32)
    mm = torch.full((E - 1,), Mcap, device=device, dtype=torch.int32)
    with pytest.raises(AssertionError):
        matmul_ogs(x, w, bias, masked_m=mm)


@pytest.mark.parametrize("free_replay", [False, True])
def test_masked_batched_matmul_reusable_across_occupancies(free_replay, device):
    """High -> empty -> low -> high occupancy must not leak stale output.

    The masked tile set changes between calls while the buffers stay put, which
    is exactly the CUDA-graph replay pattern the contract is designed for.
    """
    torch.manual_seed(20260918)
    E, Mcap, K, N = 8, 512, 1024, 1024
    x = torch.randn(E, Mcap, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) * 0.02
    bias = torch.zeros(E, N, device=device, dtype=torch.float32)
    dense = matmul_ogs(x, w, bias)

    for valid in ([Mcap] * E, [0] * E, [5] * E, [(i * 13) % Mcap for i in range(E)], [Mcap] * E):
        mm = torch.tensor(valid, device=device, dtype=torch.int32)
        out = matmul_ogs(x, w, bias, masked_m=mm)
        for e in range(E):
            v = valid[e]
            if v == 0:
                continue
            assert_close(out[e, :v], dense[e, :v], maxtol=2e-2, rmstol=2e-2,
                         description=f"valid={valid[:3]}... expert {e}", verbose=False)
