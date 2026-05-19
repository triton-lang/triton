import pytest

from triton._internal_testing import is_hip_cdna4
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout


def _dot_op_layout(op_idx, k_width):
    mfma = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    return ttgl.DotOperandLayout(operand_index=op_idx, parent=mfma, k_width=k_width)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("BM, BK, k_width, elem_bytes", [(128, 64, 8, 2), (64, 128, 8, 2), (128, 128, 8, 2),
                                                         (128, 128, 16, 1)])
@pytest.mark.parametrize("op_idx", [0, 1])
def test_compute_efficient_padded_shared_layout_constructs(BM, BK, k_width, elem_bytes, op_idx):
    dot_op = _dot_op_layout(op_idx, k_width)
    shape = [BM, BK] if op_idx == 0 else [BK, BM]
    layout = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(dot_op, shape, elem_bytes)
    assert isinstance(layout, PaddedSharedLayout)
    interval, padding = layout.interval_padding_pairs[0]
    assert interval > 0
    assert padding > 0
    assert layout.shape == shape
    assert len(layout.offset_bases) > 0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
def test_compute_efficient_padded_shared_layout_invalid_returns_none():
    dot_op = _dot_op_layout(op_idx=0, k_width=8)

    # k_width=2 is outside {4, 8, 16}; AMDMFMALayout's instr_shape passes the
    # mfma_non_k_dim check, so this exercises the k_width gate.
    bad_kwidth_dot_op = _dot_op_layout(op_idx=0, k_width=2)
    assert ttgl.amd.cdna4.compute_efficient_padded_shared_layout(bad_kwidth_dot_op, [128, 64], elem_bytes=2) is None

    # elem_bytes=4 (e.g. fp32) is outside {1, 2}.
    assert ttgl.amd.cdna4.compute_efficient_padded_shared_layout(dot_op, [128, 64], elem_bytes=4) is None
