import pytest
import torch

import triton
from triton._internal_testing import is_hip_cdna4
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("BM, BK, k_width, elem_bytes", [(128, 64, 8, 2), (64, 128, 8, 2), (128, 128, 8, 2),
                                                         (128, 128, 16, 1)])
@pytest.mark.parametrize("op_idx", [0, 1])
def test_compute_efficient_padded_shared_layout_roundtrip(BM, BK, k_width, elem_bytes, op_idx):
    """Allocate shared memory using the computed layout, do a HBM->LDS->register
    ->HBM roundtrip, and verify the data survives unchanged. Confirms the
    layout's bases produce a valid bank-conflict-free mapping the hardware
    accepts."""

    @gluon.jit
    def kernel(in_ptr, out_ptr,  #
               BM: ttgl.constexpr, BK: ttgl.constexpr, k_width: ttgl.constexpr, elem_bytes: ttgl.constexpr,
               op_idx: ttgl.constexpr):
        shared: ttgl.constexpr = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(
            op_idx=op_idx, k_width=k_width, mfma_non_k_dim=16, k_dim=BK, non_k_dim=BM, elem_bytes=elem_bytes,
            is_k_contig=True)
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [THREADS_PER_WARP // 8, 8], [4, 1], [1, 0])

        smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [BM, BK], shared)
        offs_m = ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, blocked))[:, None]
        offs_k = ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, blocked))[None, :]
        offs = offs_m * BK + offs_k

        cdna4_async_copy.buffer_load_to_shared(smem, in_ptr, offs)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(0)
        data = smem.load(blocked)
        ttgl.amd.cdna4.buffer_store(stored_value=data, ptr=out_ptr, offsets=offs)

    dtype = torch.float8_e4m3fnuz if elem_bytes == 1 else torch.float16
    inp = torch.randn(BM, BK, device="cuda", dtype=torch.float32).to(dtype)
    out = torch.empty_like(inp)
    kernel[(1, )](inp, out, BM, BK, k_width, elem_bytes, op_idx, num_warps=4)
    torch.testing.assert_close(out, inp, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
def test_compute_efficient_padded_shared_layout_invalid_returns_none():
    """Constraint violations return None so callers can fall back to a
    hand-coded layout instead of getting an opaque exception."""
    bad_mfma_dim = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=8, mfma_non_k_dim=64,
                                                                         k_dim=64, non_k_dim=128, elem_bytes=2,
                                                                         is_k_contig=True)
    assert bad_mfma_dim is None

    bad_kwidth = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=2, mfma_non_k_dim=16, k_dim=64,
                                                                       non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert bad_kwidth is None

    bad_elem_bytes = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=8, mfma_non_k_dim=16,
                                                                           k_dim=64, non_k_dim=128, elem_bytes=4,
                                                                           is_k_contig=True)
    assert bad_elem_bytes is None
