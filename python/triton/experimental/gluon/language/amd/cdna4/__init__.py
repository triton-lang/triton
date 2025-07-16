from __future__ import annotations
from typing import TYPE_CHECKING

from triton.experimental.gluon.language._core import builtin

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["get_amd_mfma_layout"]


@builtin
def get_amd_mfma_layout(version, warps_per_cta, tiles_per_warp, m_dim, n_dim, transposed, ctas_per_cga, cta_split_num,
                        cta_order, elem_type_width, _semantic: GluonSemantic = None):
    # TBD
    return _semantic.builder.get_amd_mfma_layout(version, warps_per_cta, tiles_per_warp, m_dim, n_dim, transposed,
                                                 ctas_per_cga, cta_split_num, cta_order, elem_type_width)
