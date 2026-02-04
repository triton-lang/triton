"""
Cluster Launch Control (CLC) for Blackwell (SM100+) dynamic persistent kernels.

CLC enables hardware-based dynamic work scheduling where running workers can
cancel not-yet-launched clusters and take over their work via the
clusterlaunchcontrol.try_cancel instruction.
"""

from triton.experimental.gluon.language._core import builtin, tensor
from triton.language.core import _aggregate as aggregate
import triton.language as tl

__all__ = [
    "try_cancel",
    "load_result",
    "CLCResult",
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
def load_result(result, _semantic=None):
    """
    Load the CLC response from shared memory into registers.

    Args:
        result (shared_memory_descriptor): The CLC response buffer

    Returns:
        CLCResult: Object with is_canceled() and get_first_ctaid(dim) methods
    """
    lo, hi = _semantic.builder.create_clc_load_result(result.handle)
    return CLCResult(tensor(lo, tl.int64), tensor(hi, tl.int64))


@aggregate
class CLCResult:
    """CLC response loaded into registers. Query without re-reading memory."""

    lo: tl.tensor
    hi: tl.tensor

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    @builtin
    def is_canceled(self, _semantic=None):
        """
        Check if the CLC response indicates a successful cancellation.

        Returns:
            tensor: True if a cluster was successfully canceled, False otherwise
        """
        handle = _semantic.builder.create_clc_is_canceled(self.lo.handle, self.hi.handle)
        return tensor(handle, tl.int1)

    @builtin
    def get_first_ctaid(self, dim, _semantic=None):
        """
        Get the first CTA ID coordinate of the canceled cluster.

        Args:
            dim (int): Dimension to get (0=x, 1=y, 2=z)

        Returns:
            tensor: The CTA ID coordinate value for the specified dimension
        """
        handle = _semantic.builder.create_clc_get_first_ctaid(self.lo.handle, self.hi.handle, dim)
        return tensor(handle, tl.int32)
