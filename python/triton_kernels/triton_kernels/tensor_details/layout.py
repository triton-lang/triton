from .layout_details.base import Layout
from .layout_details.blackwell_scale import BlackwellMXScaleLayout
from .layout_details.hopper_scale import HopperMXScaleLayout
from .layout_details.hopper_value import HopperMXValueLayout
from .layout_details.strided import StridedLayout
from ..target_info import cuda_capability_geq

__all__ = [
    "Layout",
    "BlackwellMXScaleLayout",
    "HopperMXScaleLayout",
    "HopperMXValueLayout",
    "StridedLayout",
]


def make_default_matmul_mxfp4_w_layout(mx_axis: int):
    if cuda_capability_geq(10):
        return StridedLayout, dict()
    elif cuda_capability_geq(9):
        return HopperMXValueLayout, {"mx_axis": mx_axis}
    else:
        return StridedLayout, dict()


def make_default_matmul_mxfp4_w_scale_layout(mx_axis: int, num_warps: int = 8):
    if cuda_capability_geq(10):
        return BlackwellMXScaleLayout, dict()
    elif cuda_capability_geq(9):
        return HopperMXScaleLayout, {"mx_axis": mx_axis, "num_warps": num_warps}
    else:
        return StridedLayout, dict()
