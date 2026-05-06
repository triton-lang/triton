from __future__ import annotations

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["arrive", "wait", "barrier", "cta_rank"]


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


@builtin
def cta_rank(_semantic=None):
    """
    Return the linear rank of the current CTA within its cluster.

    The result is in the range [0, num_ctas). For single-CTA launches this is
    zero.
    """
    handle = _semantic.builder.create_cluster_cta_rank()
    return ttgl.tensor(handle, ttgl.int32)
