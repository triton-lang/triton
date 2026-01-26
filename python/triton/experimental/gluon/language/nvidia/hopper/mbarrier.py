from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = ["allocate_mbarrier", "arrive", "expect", "init", "invalidate", "MBarrierLayout", "wait"]


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
