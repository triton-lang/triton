import pytest
import triton
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_basic_fp16():
    """Smoke test: valid fp16 inputs produce a PaddedSharedLayout."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=8, mfma_non_k_dim=16, k_dim=64,
                                                                   non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert layout is not None
    assert isinstance(layout, PaddedSharedLayout)
    assert layout.shape == [128, 64]


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_invalid_mfma_dim_returns_none():
    """mfma_non_k_dim outside {16, 32} returns None instead of raising."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=8, mfma_non_k_dim=64, k_dim=64,
                                                                   non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert layout is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_invalid_kwidth_returns_none():
    """k_width outside {4, 8, 16} returns None."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=2, mfma_non_k_dim=16, k_dim=64,
                                                                   non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert layout is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_invalid_elem_bytes_returns_none():
    """elem_bytes outside {1, 2} returns None (e.g. fp32 not supported)."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=8, mfma_non_k_dim=16, k_dim=64,
                                                                   non_k_dim=128, elem_bytes=4, is_k_contig=True)
    assert layout is None


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_operand_b():
    """op_idx=1 (operand B) produces a layout with shape [k_dim, non_k_dim]."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=1, k_width=8, mfma_non_k_dim=16, k_dim=64,
                                                                   non_k_dim=128, elem_bytes=2, is_k_contig=True)
    assert layout is not None
    assert isinstance(layout, PaddedSharedLayout)
    assert layout.shape == [64, 128]


@pytest.mark.skipif(not is_hip(), reason="CDNA4 padded layout only on HIP backend")
def test_fp8():
    """fp8 (elem_bytes=1, k_width=16) produces a valid layout."""
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=0, k_width=16, mfma_non_k_dim=16, k_dim=128,
                                                                   non_k_dim=128, elem_bytes=1, is_k_contig=True)
    assert layout is not None
    assert isinstance(layout, PaddedSharedLayout)
