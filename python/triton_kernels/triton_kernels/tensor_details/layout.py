from .layout_details.base import Layout
from .layout_details.blackwell_scale import BlackwellMXScaleLayout
from .layout_details.blackwell_scale import BlackwellActMXScaleLayout
from .layout_details.blackwell_value import BlackwellMXValueLayout
from .layout_details.hopper_scale import HopperMXScaleLayout
from .layout_details.hopper_value import HopperMXValueLayout
from .layout_details.cdna4_scale import CDNA4MXScaleLayout
from .layout_details.strided import StridedLayout
from ..target_info import cuda_capability_geq, is_hip_cdna4

__all__ = [
    "Layout",
    "BlackwellMXValueLayout",
    "BlackwellMXScaleLayout",
    "HopperMXScaleLayout",
    "HopperMXValueLayout",
    "CDNA4MXScaleLayout",
    "StridedLayout",
    "BlackwellActMXScaleLayout",
]


def make_default_matmul_mxfp4_w_layout(mx_axis: int):
    if cuda_capability_geq(10):
        return BlackwellMXValueLayout()
    elif cuda_capability_geq(9):
        return HopperMXValueLayout(mx_axis=mx_axis, mma_version=3)
    else:
        return StridedLayout(-2)


def make_default_matmul_mxfp4_w_scale_layout(mx_axis: int, num_warps: int = 8):
    if is_hip_cdna4():
        return CDNA4MXScaleLayout()
    else:
        if cuda_capability_geq(10):
            return BlackwellMXScaleLayout()
        elif cuda_capability_geq(9):
            return HopperMXScaleLayout(mx_axis=mx_axis, num_warps=num_warps)

    return StridedLayout(-2)


def make_default_matmul_mxfp8_act_scale_layout(ragged_metadata):
    if cuda_capability_geq(10):
        return BlackwellActMXScaleLayout(ragged_metadata)
    return StridedLayout(-2)
