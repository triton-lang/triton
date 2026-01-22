from __future__ import annotations


class warp_pipeline_stage:
    """
    Marks the end of a warp-pipeline stage inside a Gluon kernel.

    When used inside @gl.kernel, exiting the `with` block inserts a
    warp-pipeline border in the semantic IR. During lowering, these borders
    define pipeline clusters (scf.execute_region), drive dependency analysis,
    and determine where conditional and cluster-scope barriers are required.

    The optional string label (e.g., "load", "compute") is attached to the
    border op and may be used by downstream passes for diagnostics.

    Example:
        @gl.kernel
        def gemm(K: gl.i32):
            one = gl.const_i32(1)
            offs_a = ...

            for k in gl.range(0, K, one):

                # Stage 0: prefetch tiles
                with amd.warp_pipeline_stage("load"):
                    a = gl.amd.buffer_load(a_ptr, offs_a)
                    b = gl.amd.buffer_load(b_ptr, offs_b)

                # Stage 1: prepare MFMA operands
                with amd.warp_pipeline_stage("prep"):
                    a_tile = a.load(layout=...)
                    b_tile = b.load(layout=...)

                # Stage 2: compute
                with amd.warp_pipeline_stage("compute"):
                    acc = gl.amd.mfma(a_tile, b_tile, acc)
                    offs_a += strideA
                    offs_b += strideB

    """

    __slots__ = ("label", "_semantic", "str_attr")

    def __init__(self, label=None, **_internal):
        self.label = getattr(label, "value", None)
        self._semantic = _internal.get("_semantic", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        if self._semantic is None:
            return False
        if self.label is None:
            attr = "cluster"
        else:
            attr = self.label
        self._semantic.builder.create_warp_pipeline_border(attr)

        return False
