from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language import _core as ttgl
from ._layouts import AMDMFMALayout, AMDRotatingSharedLayout
from . import cdna3, cdna4
if TYPE_CHECKING:
    from .._semantic import GluonSemantic

from .._core import builtin, _unwrap_if_constexpr, distributed_type

@builtin
def in_thread_transpose(value, layout, _semantic: GluonSemantic = None):
    layout = _unwrap_if_constexpr(layout)
    type = distributed_type(value.type.element_ty, value.shape, layout)
    builder = _semantic.builder
    handle = builder.create_in_thread_transpose(type.to_ir(builder), value.handle)
    return ttgl.tensor(handle, type)

__all__ = ["AMDMFMALayout", "AMDRotatingSharedLayout","cdna3", "cdna4", "in_thread_transpose"]
