"""Linear asynchronous bulk copies (without a tensor map)."""

from ..._core import builtin, uint32, uint64, _unwrap_if_constexpr
from ... import _core as ttgl
from ..._semantic import _check

__all__ = ["async_load"]


@builtin
def async_load(smem, pointer, num_bytes, barrier, pred=True, multicast=False, _semantic=None):
    """Copy contiguous global-memory ranges to shared memory asynchronously.

    Requires NVIDIA compute capability 9.0 or newer and PTX 8.6 or newer.
    ``pointer`` and ``pred`` must be scalars. Each CTA copies a contiguous
    physical shared-memory range; both addresses must be 16-byte aligned.
    Layouts are interpreted through their canonical linear mapping. The copy
    preserves physical byte order: it does not swizzle or gather source data.
    Views with holes or intervening padding are not supported.

    Source ranges use full-capacity strides in increasing CTA-id order, with
    replicated CTA bits omitted. Tile ``i`` starts at byte offset
    ``i * smem.nbytes_per_cta`` from ``pointer``, even for partial copies.
    Only ``num_bytes`` bytes are read from each range. Replicated CTAs read
    the same range. With ``multicast=True``, one
    representative CTA issues each copy to its replicas and their barriers.

    ``num_bytes`` is a positive scalar integer multiple of 16, no larger than
    ``smem.nbytes_per_cta``. Only that prefix is copied. Compiler dependencies
    and ConSan conservatively cover the entire destination view, so use a
    sliced view when independently reusing other parts of an allocation.

    ``barrier`` must contain a separate completion barrier in each CTA,
    as returned by ``mbarrier.allocate_mbarrier()``. Shared logical barriers
    spanning multiple CTAs are not supported by this instruction.
    Initialize ``barrier`` and call ``mbarrier.expect(barrier, num_bytes)``
    before issuing the copy. Wait on its current phase before reading or
    reusing the destination, modifying the source, or invalidating the barrier.
    When predicating the copy, predicate the expectation consistently. The
    wait makes the copied bytes visible to ordinary shared-memory accesses.
    Before a multicast write, synchronize the cluster after prior accesses to
    its destination, including reads by other CTAs in a previous iteration.
    A per-CTA completion wait does not join the other CTAs' readers.
    Global writes made through the generic proxy must also be made visible to
    the asynchronous proxy before the copy (see PTX ``fence.proxy.async``).
    GSan records the source read at issue time; it does not currently detect
    all source modifications that race with the pending asynchronous copy.

    Args:
        smem: Destination shared-memory descriptor.
        pointer: Scalar global-memory pointer.
        num_bytes: Scalar integer byte count, possibly determined at runtime.
        barrier: Completion mbarrier.
        pred: Scalar predicate. Defaults to True.
        multicast: Multicast to replicated CTAs. Defaults to False.
    """
    num_bytes = _semantic.to_tensor(num_bytes)
    _check(num_bytes.type.is_int() and num_bytes.type.primitive_bitwidth <= 32,
           lambda: "num_bytes must be a scalar integer of at most 32 bits")
    num_bytes = _semantic.cast(num_bytes, uint32)
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
        positive = num_bytes.__gt__(0, _semantic=_semantic)
        multiple = num_bytes.__mod__(16, _semantic=_semantic).__eq__(0, _semantic=_semantic)
        in_bounds = num_bytes.__le__(smem.nbytes_per_cta, _semantic=_semantic)
        valid_size = positive.__and__(multiple, _semantic=_semantic).__and__(in_bounds, _semantic=_semantic)
        ttgl.device_assert(disabled.__or__(valid_size, _semantic=_semantic),
                           "bulk.async_load byte count must be a positive multiple of 16 within the destination view",
                           _semantic=_semantic)
    _semantic.builder.create_async_bulk_copy_global_to_local(smem.handle, pointer.handle,
                                                             num_bytes.handle, barrier.handle, pred.handle,
                                                             _unwrap_if_constexpr(multicast))
