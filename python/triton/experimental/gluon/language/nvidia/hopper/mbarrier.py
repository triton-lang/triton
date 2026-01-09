from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from triton.experimental.gluon._runtime import jit
from ..._core import _unwrap_if_constexpr, builtin
from . import cluster

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "sync_cluster_init",
    "fence_init_release_cluster",
    "init",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


@builtin
def expect(mbarrier, bytes_per_cta=None, pred=True, _semantic=None):
    """
    Expect a specific number of bytes being copied. When they are copied, the barrier is signaled.

    Args:
        mbarrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        bytes_per_cta (int): Expected byte count per CTA.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    pred = _semantic.to_tensor(pred)
    bytes_per_cta = _unwrap_if_constexpr(bytes_per_cta)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes_per_cta, pred.handle)


@builtin
def arrive(mbarrier, *, count=1, pred=True, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = _unwrap_if_constexpr(count)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle)


@builtin
def fence_init_release_cluster(_semantic=None):
    """
    Fence that makes prior mbarrier initialization visible across the CTA cluster.

    Needs to be called together with cluster.arrive(relaxed=True) and cluster.wait.
    """
    _semantic.builder.create_fence_mbarrier_init_release_cluster()


@jit
def sync_cluster_init():
    """
    Ensure mbarrier initialization is visible across the CTA cluster.
    """
    fence_init_release_cluster()
    cluster.arrive(relaxed=True)
    cluster.wait()
