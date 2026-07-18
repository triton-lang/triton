from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
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
def arrive(mbarrier, *, count=1, cta_mask=0, pred=True, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    When ``cta_mask`` is non-zero, the arrive is multicast across the cluster.
    Each bit set in the mask identifies a CTA ID dimension to multicast
    along. CTA IDs ``a`` and ``b`` belong to the same equivalence class iff
    ``a & ~cta_mask == b & ~cta_mask``; all CTAs in a class multicast to each
    other. Multicast requires ``num_ctas > 1``, ``0 < cta_mask <= num_ctas - 1``,
    and the barrier must have the identity CGA layout ``[[1], [2], ...]``. The
    default value of ``cta_mask`` is 0 (no multicast).

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        cta_mask (int): CTA broadcast dimension bits (see above). Defaults
            to 0 (no multicast). Must satisfy ``0 < cta_mask <= num_ctas - 1``
            when non-zero.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = _unwrap_if_constexpr(count)
    cta_mask = _unwrap_if_constexpr(cta_mask)
    if not isinstance(cta_mask, int) or isinstance(cta_mask, bool):
        raise TypeError(f"cta_mask must be an int, got {type(cta_mask).__name__}")
    if cta_mask:
        num_ctas = _semantic.builder.options.num_ctas
        if cta_mask < 0:
            raise ValueError(f"cta_mask must be positive, got {cta_mask}")
        if cta_mask > num_ctas - 1:
            raise ValueError(f"cta_mask must be <= num_ctas - 1 ({num_ctas - 1}), got {cta_mask}")
        sm = int(_semantic.builder.options.arch.removeprefix("sm"))
        if sm < 107:
            raise ValueError(f"multicast arrive requires Rubin (sm_107+), got sm_{sm}")
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, cta_mask, pred.handle)


@builtin
def fence_init_release_cluster(_semantic=None):
    """
    Fence that makes prior mbarrier initialization visible across the CTA cluster.

    Needs to be called together with cluster.barrier(relaxed=True).
    """
    _semantic.builder.create_fence_mbarrier_init_release_cluster()
