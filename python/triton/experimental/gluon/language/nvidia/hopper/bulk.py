"""Linear asynchronous bulk copies (without a tensor map)."""

from ..._core import builtin, uint64, _unwrap_if_constexpr
from ... import _core as ttgl
from ..._semantic import _check

__all__ = ["async_load"]


@builtin
def async_load(smem, pointer, num_bytes, barrier, pred=True, multicast=False, _semantic=None):
    """Copy contiguous global-memory ranges to shared memory asynchronously.

    Requires compute capability 9.0+ and PTX 8.6+. ``pointer`` and ``pred``
    are scalars; both copy addresses must be 16-byte aligned. Each CTA copies
    a contiguous physical range, preserving byte order without swizzling or
    gathering. Layouts use their canonical linear mapping; views with holes
    or intervening padding are unsupported.

    ``num_bytes`` is a positive compile-time multiple of 16, at most
    ``smem.nbytes_per_cta``. Source tile ``i`` starts at byte offset
    ``i * smem.nbytes_per_cta`` from ``pointer``, even for partial copies.
    Tiles follow increasing CTA-id order with replicated bits omitted:
    replicas read the same range, or one representative multicasts to them
    and their barriers when ``multicast=True``.

    Only the ``num_bytes`` prefix is copied, but compiler dependencies and
    ConSan cover the entire destination view. Use sliced views to reuse
    other parts of the allocation independently.

    Use separate completion barriers in each CTA, as returned by
    ``mbarrier.allocate_mbarrier()``; shared logical barriers are unsupported.
    Initialize the barrier, then call ``mbarrier.expect(barrier, num_bytes)``
    with the same predicate as the copy. Wait on its current phase before
    reading or reusing the destination, modifying the source, or invalidating
    the barrier. This wait makes the copy visible to ordinary shared accesses.
    Before a multicast write, join the cluster after prior destination
    accesses, including other CTAs' reads from a previous iteration;
    per-CTA completion waits do not join those readers.

    Make generic-proxy global writes visible to the asynchronous proxy
    before copying (see PTX ``fence.proxy.async``). GSan records source reads
    at issue time and may miss modifications racing with the pending copy.

    Args:
        smem: Destination shared-memory descriptor.
        pointer: Scalar global-memory pointer.
        num_bytes: Compile-time integer byte count.
        barrier: Completion mbarrier.
        pred: Scalar predicate. Defaults to True.
        multicast: Multicast to replicated CTAs. Defaults to False.
    """
    num_bytes = _unwrap_if_constexpr(num_bytes)
    _check(isinstance(num_bytes, int), lambda: "num_bytes must be a compile-time integer")
    pointer = _semantic.to_tensor(pointer)
    pred = _semantic.to_tensor(pred)
    _check(pointer.type.is_ptr(), lambda: "pointer must be a scalar global-memory pointer")
    _check(pred.type.is_bool(), lambda: "pred must be a scalar boolean")
    if _semantic.builder.options.enable_iisan:
        address = _semantic.cast(pointer, uint64)
        aligned = address.__mod__(16, _semantic=_semantic).__eq__(0, _semantic=_semantic)
        disabled = pred.__invert__(_semantic=_semantic)
        ttgl.device_assert(disabled.__or__(aligned, _semantic=_semantic),
                           "bulk.async_load source must be 16-byte aligned", _semantic=_semantic)
    _semantic.builder.create_async_bulk_copy_global_to_local(smem.handle, pointer.handle, num_bytes, barrier.handle,
                                                             pred.handle, _unwrap_if_constexpr(multicast))
