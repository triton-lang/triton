from __future__ import annotations


class warp_pipeline_stage:
    """
    Marks a warp-pipeline stage inside a Gluon kernel.

	When used inside @gl.kernel, each with amd.warp_pipeline_stage(...) block
	semantically defines a distinct stage of a warp pipeline. All operations
	inside the block belong to the same pipeline cluster and are intended to
	execute as a unit relative to other stages.

	The optional string label (e.g., "load", "compute") names the pipeline
	stage for identification and diagnostics, without affecting program
	semantics.

	An optional integer priority may be specified to express the relative
	scheduling priority of the warp the stage belongs to. The priority applies
	to the entire cluster. Valid values range from 0 (lowest) to 3 (highest)
    as it's lowered to the operand of `s_setprio`. If unspecified, priority
    resets to zero when any other stage in the loop uses explicit priority;
    otherwise no priority instruction is emitted.
    N.B., This is a performance hint to the hardware scheduler, and its effect
	may vary depending on the dynamic interaction of instruction streams
	across different warps. It is optional and should be used judiciously,
	only when explicit scheduling guidance is beneficial.

    Example: (only to show how to use, this example is not supposed to
    represent the optimal way.)

    @gl.kernel
    ...

    for k in gl.range(0, K, one):

        # Stage 0: prefetch tiles
        with amd.warp_pipeline_stage("load", priority=3):
            a = gl.amd.buffer_load(a_ptr, offs_a)
            b = gl.amd.buffer_load(b_ptr, offs_b)

        # Stage 1: prepare MFMA operands
        with amd.warp_pipeline_stage("prep"):
            a_tile = a.load(layout=...)
            b_tile = b.load(layout=...)

        # Stage 2: compute
        with amd.warp_pipeline_stage("compute", priority=0):
            acc = gl.amd.mfma(a_tile, b_tile, acc)
            offs_a += strideA
            offs_b += strideB
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
