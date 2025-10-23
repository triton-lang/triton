from triton.language.core import _unwrap_if_constexpr
from ._layouts import AMDMFMALayout, AMDWMMALayout


def _get_mfma_scale_layout(dot_operand_layout, shape, semantic):
    dot_operand_layout = _unwrap_if_constexpr(dot_operand_layout)
    shape = _unwrap_if_constexpr(shape)

    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    tiles_per_warp = parent.tiles_per_warp
    warps_per_cta = parent.warps_per_cta
    return semantic.builder.get_amd_mfma_scale_layout(op_idx, shape, mdim, tiles_per_warp, warps_per_cta)


def _get_wmma_scale_layout(dot_operand_layout, shape, semantic):
    dot_operand_layout = _unwrap_if_constexpr(dot_operand_layout)
    shape = _unwrap_if_constexpr(shape)

    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDWMMALayout), "Expected parent to be an instance of AMDMFMALayout"
    warps_per_cta = parent.warps_per_cta
    return semantic.builder.get_amd_wmma_scale_layout(op_idx, warps_per_cta, shape)


def _get_scale_shape(op_idx, operand, format):
    operand_shape = [s for s in operand.type.shape]
    scale_shape = operand_shape
    unpack_factor = 2 if format.value == "e2m1" else 1
    if op_idx == 0:
        k = scale_shape[-1] * unpack_factor
        scale_shape[-1] = k // 32
    else:
        k = scale_shape[-2] * unpack_factor
        scale_shape[-2] = k // 32
        scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
    return scale_shape
