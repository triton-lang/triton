from .._core import builtin
from ._layouts import AMDMFMALayout, AMDWMMALayout
from . import cdna3, cdna4
from . import rdna3, rdna4
from . import gfx1250

__all__ = ["AMDMFMALayout", "AMDWMMALayout", "cdna3", "cdna4", "rdna3", "rdna4", "gfx1250", "split_warp_pipeline"]


@builtin
def split_warp_pipeline(_semantic=None):
    return _semantic.builder.create_warp_pipeline_border()
