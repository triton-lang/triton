"""
Grid Dependency Control (GDC) is a mechanism used when enabling programmatic dependent launch to launch and
synchronize grids. These APIs expose GDC to the programmer.

Programmatic dependent launch is supported on SM90 (Hopper) and beyond.
For PTX reference on grid dependency control see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol.
"""

from triton.language import core


@core.extern
def gdc_wait(_semantic=None):
    """
    GDC wait is a blocking instruction that waits for all instructions in a prior kernel to complete before continuing.
    This ensures all memory operations happening before the wait is visible to instructions after it,
    e.g. if the prior kernel writes to address "x" the new values will be visible in this kernel after the wait.

    This instruction is also safe to execute when programmatic dependent launch is disabled.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol for more details.
    """
    _semantic.grid_dependency_wait()


@core.extern
def gdc_launch_dependents(_semantic=None):
    """
    This operation when launched with programmatic dependent launch signals that
    the next program may launch once all programs in the current kernel
    call this function or complete.

    Repeated calls to this function have no effect past the first call, and the first call should be
    treated by the programmer as a hint to the runtime system to launch the next kernel.

    This instruction is also safe to execute when programmatic dependent launch is disabled.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol for more details.
    """
    _semantic.grid_dependency_launch_dependents()
