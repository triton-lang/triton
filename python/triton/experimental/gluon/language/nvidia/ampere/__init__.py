from __future__ import annotations

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._layouts import DotOperandLayout, NVMMADistributedLayout
from ..._core import builtin, _unwrap_if_constexpr
from . import async_copy, mbarrier

__all__ = ["async_copy", "mbarrier", "mma_v2"]


@builtin
def mma_v2(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    assert isinstance(a, ttgl.tensor), "a must be a tensor"
    assert isinstance(b, ttgl.tensor), "b must be a tensor"
    assert isinstance(acc, ttgl.tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, NVMMADistributedLayout), "acc must have an NVMMADistributedLayout"
    assert mma_layout.version == [2, 0], "MMA layout must have version 2.0"

    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"

    handle = _semantic.dot(a, b, acc, input_precision=input_precision, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
