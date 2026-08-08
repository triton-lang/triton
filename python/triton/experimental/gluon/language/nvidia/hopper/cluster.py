from __future__ import annotations

from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["barrier"]


@builtin
def barrier(relaxed: bool = False, _semantic=None):
    """
    Barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed arrival semantics. Defaults to
            False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_barrier(relaxed)
