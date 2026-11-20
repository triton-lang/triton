"""Grid Dependency Control (GDC) for programmatic dependent launch on SM90 and newer."""

from ..._core import builtin

__all__ = ["wait", "launch_dependents"]


@builtin
def wait(_semantic=None):
    """
    Wait for the preceding kernel to complete when using programmatic dependent launch.

    Memory operations from the preceding kernel are visible to accesses after
    this wait. Enable programmatic dependent launch with ``launch_pdl=True``
    when launching the dependent kernel.

    Requires SM90 or newer. Safe to call when programmatic dependent launch is disabled.
    """
    _semantic.grid_dependency_wait()


@builtin
def launch_dependents(_semantic=None):
    """
    Signal that dependent kernels may launch when using programmatic dependent launch.

    A dependent kernel may launch once all programs in this kernel have called
    this function or completed. This is a scheduling hint; repeated calls have
    no additional effect. Dependent kernels must call :func:`wait` before
    accessing results produced by this kernel.

    Requires SM90 or newer. Safe to call when programmatic dependent launch is disabled.
    """
    _semantic.grid_dependency_launch_dependents()
