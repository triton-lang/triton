from __future__ import annotations

from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["arrive", "wait"]


@builtin
def arrive(relaxed: bool = False, _semantic=None):
    """
    Arrive at a barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed semantics. Defaults to False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_arrive(relaxed)


@builtin
def wait(_semantic=None):
    """
    Wait for all CTAs in the cluster to arrive at the cluster barrier.
    """
    _semantic.builder.create_cluster_wait()
