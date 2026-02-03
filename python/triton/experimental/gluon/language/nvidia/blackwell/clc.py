"""
Cluster Launch Control (CLC) for Blackwell (SM100+) dynamic persistent kernels.

CLC enables hardware-based dynamic work scheduling where running workers can
cancel not-yet-launched clusters and take over their work via the
clusterlaunchcontrol.try_cancel instruction.
"""

from triton.experimental.gluon._runtime import jit
from triton.experimental.gluon.language._core import builtin, tensor
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
import triton.language as tl

__all__ = [
    "try_cancel",
    "is_canceled",
    "get_first_ctaid",
]


@builtin
def try_cancel(result, mbar, multicast=False, _semantic=None):
    """
    Issue a CLC try_cancel request to atomically cancel a pending cluster launch.
    
    Args:
        result (shared_memory_descriptor): 16-byte aligned shared memory for the response
        mbar (shared_memory_descriptor): 8-byte aligned mbarrier for completion signaling
        multicast (bool): If True, broadcast result to all CTAs in cluster
    
    Only supported on SM100+ (Blackwell).
    """
    _semantic.builder.create_clc_try_cancel(result.handle, mbar.handle, multicast)


@builtin
def is_canceled(result, _semantic=None):
    """
    Check if the CLC response indicates a successful cancellation.
    
    Args:
        result (shared_memory_descriptor): The CLC response buffer
        
    Returns:
        tensor: Non-zero if a cluster was successfully canceled, 0 otherwise
    """
    handle = _semantic.builder.create_clc_is_canceled(result.handle)
    return tensor(handle, tl.int32)


@builtin
def get_first_ctaid(result, dim, _semantic=None):
    """
    Get the first CTA ID coordinate of the canceled cluster from CLC response.
    
    Args:
        result (shared_memory_descriptor): The CLC response buffer
        dim (int): Dimension to get (0=x, 1=y, 2=z)
        
    Returns:
        tensor: The CTA ID coordinate value for the specified dimension
    """
    handle = _semantic.builder.create_clc_get_first_ctaid(result.handle, dim)
    return tensor(handle, tl.int32)


@jit
def fetch_next_tile(clc_result, clc_mbar):
    """
    High-level helper to fetch next tile via CLC.
    
    Args:
        clc_result: shared_memory_descriptor for 16-byte CLC response
        clc_mbar: shared_memory_descriptor for mbarrier
        
    Returns:
        (has_work, tile_m, tile_n): Tuple of whether work was found and tile coordinates
    """
    # Issue CLC try_cancel
    try_cancel(clc_result, clc_mbar)

    # Wait for CLC response
    mbarrier.wait(clc_mbar, 0)

    # Decode response
    has_work = is_canceled(clc_result)
    tile_m = get_first_ctaid(clc_result, 0)
    tile_n = get_first_ctaid(clc_result, 1)

    return has_work, tile_m, tile_n
