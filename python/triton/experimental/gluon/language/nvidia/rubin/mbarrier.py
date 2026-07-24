from triton.language import core as tl

from ..._core import _unwrap_if_constexpr, builtin
from ..hopper.mbarrier import (
    MBarrierLayout,
    allocate_mbarrier,
    expect,
    init,
    invalidate,
)

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "init",
    "invalidate",
    "MBarrierLayout",
    "test_wait",
    "wait",
]


@builtin
def wait(mbarrier, phase, pred=True, conditional=False, deps=(), _semantic=None):
    """
    Wait until the mbarrier object completes the requested phase.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase/parity value to wait for.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
        conditional (bool): If True, wait on the conditional phase used by
            TMA report-validity operations. Defaults to False.
        deps (Sequence[shared_memory_descriptor]): Dependent allocations barrier is waiting on. Used to track liveness of dependent allocations. Defaults to ().
    """
    conditional = _unwrap_if_constexpr(conditional)
    phase = _semantic.to_tensor(phase)
    pred = _semantic.to_tensor(pred)
    deps = [x.handle for x in deps]
    _semantic.builder.create_mbarrier_wait(mbarrier.handle, phase.handle, pred.handle, deps, conditional)


@builtin
def test_wait(mbarrier, phase, pred=True, conditional=False, _semantic=None):
    """
    Test an mbarrier phase once without blocking.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to test.
        phase (int): The phase/parity value to test.
        pred (bool): Predicate. Operation is skipped if predicate is False.
            Defaults to True.
        conditional (bool): If True, test the conditional phase used by TMA
            report-validity operations. Defaults to False.

    Returns:
        tensor: Scalar int32 tensor containing 1 if the requested phase has
            completed, otherwise 0.
    """
    conditional = _unwrap_if_constexpr(conditional)
    phase = _semantic.to_tensor(phase)
    pred = _semantic.to_tensor(pred)
    handle = _semantic.builder.create_mbarrier_test_wait(mbarrier.handle, phase.handle, pred.handle, conditional)
    return _semantic.tensor(handle, tl.int32)


@builtin
def arrive(mbarrier, *, count=1, pred=True, from_cta=None, multicast_cta=0, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    When ``multicast_cta`` is non-zero, the arrive is multicast across the cluster.
    Each bit set in the mask identifies a CTA ID dimension to multicast
    along. CTA IDs ``a`` and ``b`` belong to the same equivalence class iff
    ``a & ~multicast_cta == b & ~multicast_cta``; all CTAs in a class multicast to each
    other. Multicast requires ``num_ctas > 1``, ``0 < multicast_cta <= num_ctas - 1``,
    and the barrier must have the identity CGA layout ``[[1], [2], ...]``. The
    default value of ``multicast_cta`` is 0 (no multicast).

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        from_cta (int, optional): Mask of CTA-ID bits preserved when routing the arrival, in
            ``[0, num_ctas - 1]``. Defaults to ``num_ctas - 1``, which arrives from each CTA to itself; ``0``
            routes from CTA 0 to every CTA. A non-identity mask cannot be combined with multicast.
        multicast_cta (int): CTA broadcast dimension bits (see above). Defaults
            to 0 (no multicast). Must satisfy ``0 < multicast_cta <= num_ctas - 1``
            when non-zero.
    """
    count = _unwrap_if_constexpr(count)
    from_cta = _unwrap_if_constexpr(from_cta)
    multicast_cta = _unwrap_if_constexpr(multicast_cta)
    if not isinstance(multicast_cta, int) or isinstance(multicast_cta, bool):
        raise TypeError(f"multicast_cta must be an int, got {type(multicast_cta).__name__}")
    if multicast_cta:
        num_ctas = _semantic.builder.options.num_ctas
        if multicast_cta < 0:
            raise ValueError(f"multicast_cta must be positive, got {multicast_cta}")
        if multicast_cta > num_ctas - 1:
            raise ValueError(f"multicast_cta must be <= num_ctas - 1 ({num_ctas - 1}), got {multicast_cta}")
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle, from_cta, multicast_cta)
