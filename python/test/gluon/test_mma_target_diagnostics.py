import pytest

from triton.experimental.gluon._target import target_context
from triton.experimental.gluon.language import NVMMADistributedLayout


class RecordingBuilder:
    def get_mma_layout(self, *args):
        return args


def test_nv_mma_volta_layout_rejected_on_sm90():
    layout = NVMMADistributedLayout(
        version=[1, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]
    )
    with target_context("cuda:90"):
        with pytest.raises(ValueError, match="only supported on Volta"):
            layout._to_ir(RecordingBuilder())


def test_nv_mma_volta_layout_remains_valid_on_sm70():
    layout = NVMMADistributedLayout(
        version=[1, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]
    )
    with target_context("cuda:70"):
        result = layout._to_ir(RecordingBuilder())
    assert result == ([1, 0], [4, 1], [], [16, 8])
