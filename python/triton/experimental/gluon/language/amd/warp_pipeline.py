from __future__ import annotations


class warp_pipeline_stage:
    """
    Marks a warp-pipeline stage inside a Gluon kernel.

    Within a ``@gluon.jit`` kernel, each ``with gl.amd.warp_pipeline_stage(...)``
    block defines a distinct stage of a warp pipeline. All operations inside
    the block belong to the same pipeline cluster and execute as a unit
    relative to other stages.

    The optional string ``label`` (for example, ``"load"`` or ``"compute"``)
    identifies the stage in diagnostics without affecting program semantics.

    ``priority`` is an optional integer that expresses the scheduling priority
    of the warp that executes the stage. The priority applies to the entire
    cluster. Valid values range from 0 (lowest) to 3 (highest), matching the
    operand range of ``s_setprio``. If omitted, the priority resets to zero
    when another stage in the loop uses an explicit priority; otherwise, no
    priority instruction is emitted.
    N.B., This is a performance hint to the hardware scheduler. Its effect
    depends on the dynamic interaction between warp instruction streams
    across different warps. It is optional and should be used judiciously,
    only when explicit scheduling guidance is beneficial.

    The following schematic shows the stage boundaries; it is not intended as
    an optimal kernel.

    **Example**

    .. code-block:: python

        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl

        @gluon.jit
        def warp_pipelined_kernel(a_ptr, b_ptr, c_ptr, K: gl.constexpr):
            acc = 0.0

            for k in range(0, K):
                # Stage 0: prefetch tiles.
                with gl.amd.warp_pipeline_stage("load", priority=3):
                    a = gl.load(a_ptr + k)
                    b = gl.load(b_ptr + k)

                # Stage 1: prepare MFMA operands.
                with gl.amd.warp_pipeline_stage("prep"):
                    a_tile = a  # Convert to the required dot-operand layout.
                    b_tile = b  # Convert to the required dot-operand layout.

                # Stage 2: compute.
                with gl.amd.warp_pipeline_stage("compute", priority=0):
                    acc += a_tile * b_tile

            gl.store(c_ptr, acc)
    """

    __slots__ = ("label", "priority", "_semantic")

    def __init__(self, label=None, *, priority: int | None = None, **_internal):
        self.label = getattr(label, "value", None)
        if priority is not None:
            assert priority > -1 and priority < 4, "priority should be 0 to 3."
        self.priority = priority
        self._semantic = _internal.get("_semantic", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        if self._semantic is None:
            return False
        marker = self.label if self.label is not None else "cluster"
        prio = self.priority if self.priority is not None else -1
        self._semantic.builder.create_warp_pipeline_border(marker, prio)
        return False
