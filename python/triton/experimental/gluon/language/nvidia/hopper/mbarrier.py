from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin, tuple

__all__ = ["allocate_mbarrier", "arrive", "expect", "init", "invalidate", "MBarrierLayout", "wait"]


@builtin
def expect(mbarrier, bytes_per_cta=None, descs=None, pred=True, _semantic=None):
    """
    Expect a specific number of bytes being copied. When they are copied, the barrier is signaled.

    It is also possible to pass a list of tensor memory descriptors to expect the bytes for.

    Args:
        mbarrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        bytes (int): Expected byte count.
        descs (List[tensor_memory_descriptor]): List of tensor memory descriptors that will be copied.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    pred = _semantic.to_tensor(pred)
    bytes_per_cta = _unwrap_if_constexpr(bytes_per_cta)
    descs = _unwrap_if_constexpr(descs)
    assert descs is None or bytes_per_cta is None, "Only one of bytes_per_cta or descs can be provided"
    assert descs is None or isinstance(descs, tuple)
    if bytes_per_cta is None:
        bytes_per_cta = sum(desc.nbytes_per_cta for desc in descs)
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
