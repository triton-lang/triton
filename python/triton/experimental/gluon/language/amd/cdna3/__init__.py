from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language._core import builtin

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["get_amd_mfma_layout"]


@builtin
def get_amd_mfma_layout(version, tiles_per_warp, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape,
                        transposed, elem_type_width, _semantic: GluonSemantic = None):
    return _semantic.builder.get_amd_mfma_layout(version, tiles_per_warp, warps_per_cta, ctas_per_cga, cta_split_num,
                                                 cta_order, instr_shape, transposed, elem_type_width)
