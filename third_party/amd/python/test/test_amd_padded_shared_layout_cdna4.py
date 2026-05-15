import pytest

from triton._internal_testing import is_hip_cdna4
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("BM, BK, k_width, elem_bytes", [(128, 64, 8, 2), (64, 128, 8, 2), (128, 128, 8, 2),
                                                         (128, 128, 16, 1)])
@pytest.mark.parametrize("op_idx", [0, 1])
def test_compute_efficient_padded_shared_layout_constructs(BM, BK, k_width, elem_bytes, op_idx):
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(op_idx=op_idx, k_width=k_width, mfma_non_k_dim=16,
                                                                   k_dim=BK, non_k_dim=BM, elem_bytes=elem_bytes,
                                                                   is_k_contig=True)
    assert isinstance(layout, PaddedSharedLayout)
    interval, padding = layout.interval_padding_pairs[0]
    assert interval > 0
    assert padding > 0
    expected_shape = [BM, BK] if op_idx == 0 else [BK, BM]
    assert layout.shape == expected_shape
    assert len(layout.offset_bases) > 0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
def test_compute_efficient_padded_shared_layout_invalid_returns_none():
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
