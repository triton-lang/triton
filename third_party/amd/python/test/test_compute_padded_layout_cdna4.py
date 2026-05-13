import pytest
import triton


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_basic():
    """Smoke test: valid CDNA4 inputs produce a well-formed layout dict."""
    from triton._C.libtriton import amd
    r = amd.compute_padded_layout_cdna4(
        op_idx=0, k_width=8, mfma_non_k_dim=16,
        k_dim=64, non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert r is not None
    assert r["interval"] > 0
    assert r["padding"] > 0
    assert len(r["bases"]) > 0
    # Bases are 2D linear bases — each entry should be a 2-element list/tuple.
    for b in r["bases"]:
        assert len(b) == 2


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_invalid_mfma_dim():
    """Out-of-range mfma_non_k_dim must return None, not raise."""
    from triton._C.libtriton import amd
    # mfma_non_k_dim must be in {16, 32}; 64 is invalid.
    r = amd.compute_padded_layout_cdna4(
        op_idx=0, k_width=8, mfma_non_k_dim=64,
        k_dim=64, non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert r is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_invalid_kwidth():
    """k_width outside {4, 8, 16} returns None."""
    from triton._C.libtriton import amd
    r = amd.compute_padded_layout_cdna4(
        op_idx=0, k_width=2, mfma_non_k_dim=16,
        k_dim=64, non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert r is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_invalid_elem_bytes():
    """elem_bytes outside {1, 2} returns None (e.g. fp32 not supported)."""
    from triton._C.libtriton import amd
    r = amd.compute_padded_layout_cdna4(
        op_idx=0, k_width=8, mfma_non_k_dim=16,
        k_dim=64, non_k_dim=128, elem_bytes=4, is_k_contig=True)
    assert r is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_op_b():
    """op_idx=1 (operand B) also produces a valid layout."""
    from triton._C.libtriton import amd
    r = amd.compute_padded_layout_cdna4(
        op_idx=1, k_width=8, mfma_non_k_dim=16,
        k_dim=64, non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert r is not None
    assert r["interval"] > 0
    assert len(r["bases"]) > 0


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_compute_padded_layout_cdna4_fp8():
    """fp8 (elem_bytes=1) produces a valid layout."""
    from triton._C.libtriton import amd
    r = amd.compute_padded_layout_cdna4(
        op_idx=0, k_width=16, mfma_non_k_dim=16,
        k_dim=128, non_k_dim=128, elem_bytes=1, is_k_contig=True)
    assert r is not None
    assert r["interval"] > 0
