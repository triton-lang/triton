"""Linear asynchronous bulk copies (without a tensor map)."""

from ..._core import builtin, uint64, _unwrap_if_constexpr
from ... import _core as ttgl
from ..._semantic import _check

__all__ = ["async_load"]


@builtin
def async_load(smem, pointer, num_elements, barrier, pred=True, multicast=False, _semantic=None):
    """Copy contiguous global-memory ranges to shared memory asynchronously.

    Requires compute capability 9.0+ and PTX 8.6+. ``pointer`` and ``pred``
    are scalars; both copy addresses must be 16-byte aligned. Each CTA copies
    a contiguous physical range. Swizzled destination layouts are supported,
    but the source must already have the desired physical ordering: the copy
    itself does not swizzle or gather. Padded and partitioned layouts, and
    views with holes, are unsupported.

    ``num_elements`` counts destination elements across all distinct CTA tiles,
    excluding replicas, and must be a positive compile-time integer at most
    ``smem.numel``. The total is split equally between those tiles, and each
    per-CTA byte count must be a multiple of 16. If there are ``T`` distinct
    tiles, source tile ``i`` starts at element ``i * num_elements // T``,
    including for partial copies. Thus the source is one contiguous range.
    Tiles follow increasing CTA-id order with replicated bits omitted:
    replicas read the same range, or one representative multicasts to them
    and their barriers when ``multicast=True``.

    Only the per-CTA prefix is copied, but compiler dependencies and
    ConSan cover the entire destination view. Use sliced views to reuse
    other parts of the allocation independently.

    Use separate completion barriers in each CTA, as returned by
    ``mbarrier.allocate_mbarrier()``; shared logical barriers are unsupported.
    Initialize the barrier, then call ``mbarrier.expect(barrier, bytes_per_cta)``
    with the same predicate as the copy, where
    ``bytes_per_cta = num_elements * smem.nbytes_per_cta // smem.numel``.
    Wait on its current phase before reading or reusing the destination,
    modifying the source, or invalidating
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
        num_elements: Compile-time total number of destination elements.
        barrier: Completion mbarrier.
        pred: Scalar predicate. Defaults to True.
        multicast: Multicast to replicated CTAs. Defaults to False.
    """
    num_elements = _unwrap_if_constexpr(num_elements)
    _check(isinstance(num_elements, int), lambda: "num_elements must be a compile-time integer")
    num_bytes = num_elements * smem.dtype.primitive_bitwidth // 8
    _check(0 < num_elements <= smem.numel, lambda: "num_elements must be positive and fit in the destination view")
    _check(num_bytes <= 0x7fffffff, lambda: "total byte count must fit in a signed 32-bit integer")
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
