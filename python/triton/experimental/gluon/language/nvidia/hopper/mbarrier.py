from ..ampere.mbarrier import MBarrierLayout, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = ["arrive", "expect", "init", "invalidate", "MBarrierLayout", "wait"]


@builtin
def expect(mbarrier, bytes, pred=True, _semantic=None):
    """
    Expect a specific number of bytes being copied. When they are copied, the barrier is signaled.

    Args:
        mbarrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        bytes (int): Expected byte count.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    bytes = _unwrap_if_constexpr(bytes)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes, pred.handle)


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
