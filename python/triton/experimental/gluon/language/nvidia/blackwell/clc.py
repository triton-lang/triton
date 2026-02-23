"""
Cluster Launch Control (CLC) for Blackwell (SM100+) dynamic persistent kernels.

CLC enables hardware-based dynamic work scheduling where running workers can
cancel not-yet-launched clusters and take over their work via the
clusterlaunchcontrol.try_cancel instruction.
"""
from __future__ import annotations

import triton.experimental.gluon.language._core as gl
from triton.experimental.gluon.language._core import builtin, tensor, shared_memory_descriptor, base_value, base_type
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from triton._C.libtriton import ir

__all__ = [
    "try_cancel",
    "load_result",
    "clc_result",
]


@builtin
def try_cancel(result: shared_memory_descriptor, barrier, multicast=False, _semantic=None):
    """
    Issue a CLC try_cancel request to atomically cancel a pending cluster launch.

    Args:
        result (shared_memory_descriptor): 16-byte aligned shared memory for the response
        barrier (shared_memory_descriptor): 8-byte aligned mbarrier for completion signaling
        multicast (bool): If True, broadcast result to all CTAs in cluster

    Only supported on SM100+ (Blackwell).
    """
    _semantic.builder.create_clc_try_cancel(result.handle, barrier.handle, multicast)


@builtin
def load_result(src, _semantic=None):
    """
    Load the CLC response from shared memory into registers.

    Args:
        src (shared_memory_descriptor): The CLC response buffer

    Returns:
        CLCResult: Object with is_canceled() and get_first_ctaid(dim) methods
    """
    handle = _semantic.builder.create_clc_load_result(src.handle)
    return clc_result(handle)


class clc_result_type(base_type):

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_int128_ty()

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[shared_memory_descriptor, int]:
        value = clc_result(handles[cursor])
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return "clc_result"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def mangle(self) -> str:
        return "CLC"


class clc_result(base_value):
    """CLC response loaded into registers. Query without re-reading memory."""

    def __init__(self, handle):
        self.handle = handle
        self.type = clc_result_type()

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    @builtin
    def is_canceled(self, _semantic=None):
        """
        Check if the CLC response indicates a successful cancellation.

        Returns:
            tensor: True if a cluster was successfully canceled, False otherwise
        """
        handle = _semantic.builder.create_clc_is_canceled(self.handle)
        return tensor(handle, gl.int1)

    @builtin
    def program_id(self, dim, _semantic=None):
        """
        Get the Program ID of the canceled cluster.

        Args:
            dim (int): Dimension to get (0=x, 1=y, 2=z)

        Returns:
            tensor: The Program ID for the specified dimension
        """
        handle = _semantic.builder.create_clc_get_program_id(self.handle, dim)
        return tensor(handle, gl.int32)
