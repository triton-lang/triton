"""Linear asynchronous bulk copies (without a tensor map)."""

from ..._core import builtin, uint64, _unwrap_if_constexpr
from ... import _core as ttgl
from ..._semantic import _check

__all__ = ["async_load", "async_store", "store_wait"]


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
    ``smem.numel``. With multiple CTAs, including replicated layouts,
    ``num_elements`` must equal ``smem.numel``; partial copies are supported
    only with one CTA. The total is split equally between distinct tiles,
    and each per-CTA byte count must be a multiple of 16. If there are ``T``
    distinct tiles, source tile ``i`` starts at element
    ``i * smem.numel // T``. Thus the source is one contiguous range.
    Tiles follow increasing CTA-id order with replicated bits omitted:
    replicas read the same range, or one representative multicasts to them
    and their barriers when ``multicast=True``.

    For a single-CTA partial copy, only the prefix is copied, but compiler
    dependencies and ConSan cover the entire destination view. Use sliced
    views to reuse other parts of the allocation independently.

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
    _check(_semantic.builder.options.num_ctas == 1 or num_elements == smem.numel,
           lambda: "num_elements must equal smem.numel for multi-CTA copies")
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


@builtin
def async_store(pointer, smem, num_elements, pred=True, _semantic=None):
    """Copy contiguous shared-memory ranges to global memory asynchronously.

    Requires compute capability 9.0+ and PTX 8.0+. ``pointer`` and ``pred``
    are scalars. Both addresses must be 16-byte aligned. The size and shared
    layout requirements are the same as for :func:`async_load`: each CTA
    copies a contiguous physical range, without unswizzling. ``num_elements``
    is a positive compile-time total across distinct CTA tiles. Multi-CTA
    copies must fill the source view; single-CTA copies may copy a prefix.
    Tiles are packed in increasing CTA-id order with replicated bits omitted.
    Only one representative of each replicated tile issues the store.

    Each executed store commits one group to the same queue as TMA stores.
    A false predicate neither copies nor commits a group. Call
    :func:`store_wait` before modifying or reusing the shared source. To
    access the global destination or publish it with a release operation,
    first call ``store_wait(0, read_only=False)``. Issue and wait must execute
    in the same warp-specialized region. The compiler inserts the shared
    async-proxy fence needed to publish preceding shared-memory writes.

    Compiler dependencies and ConSan conservatively cover the entire source
    view, including for partial copies. GSan records the exact global write
    range at issue time; it does not track the pending transfer to completion.

    Args:
        pointer: Scalar global-memory destination pointer.
        smem: Source shared-memory descriptor.
        num_elements: Compile-time total number of source elements.
        pred: Scalar predicate. Defaults to True.
    """
    num_elements = _unwrap_if_constexpr(num_elements)
    _check(isinstance(num_elements, int), lambda: "num_elements must be a compile-time integer")
    _check(0 < num_elements <= smem.numel, lambda: "num_elements must be positive and fit in the source view")
    _check(_semantic.builder.options.num_ctas == 1 or num_elements == smem.numel,
           lambda: "num_elements must equal smem.numel for multi-CTA copies")
    num_bytes = num_elements * smem.dtype.primitive_bitwidth // 8
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
                           "bulk.async_store destination must be 16-byte aligned", _semantic=_semantic)
    _semantic.builder.create_async_bulk_copy_local_to_global(pointer.handle, smem.handle, num_bytes, pred.handle)


@builtin
def store_wait(pendings, read_only=True, _semantic=None):
    """Wait for committed bulk and TMA store groups in the current region.

    Args:
        pendings: Maximum number of committed groups allowed to remain pending.
        read_only: If True, wait only for shared-memory reads to complete, so
            the source can be reused. If False, also wait for global writes
            to complete before accessing or publishing the destination.
            Defaults to True.
    """
    pendings = _unwrap_if_constexpr(pendings)
    read_only = _unwrap_if_constexpr(read_only)
    _semantic.builder.create_async_tma_store_wait(pendings, read_only)
