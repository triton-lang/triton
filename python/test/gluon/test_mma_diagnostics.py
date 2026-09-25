import pytest

from triton.experimental.gluon import language as ttgl


def test_nv_mma_layout_rejects_rank_one():
    with pytest.raises(ValueError, match="rank >= 2"):
        ttgl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[4], instr_shape=[16, 8]
        )
