"""Linear asynchronous bulk copies (without a tensor map)."""

from ..._core import _unwrap_if_constexpr, builtin
from ..._semantic import _check

__all__ = ["async_load"]


@builtin
def async_load(smem, pointer, num_bytes, barrier, pred=True, _semantic=None):
    """Copy a contiguous global-memory range to CTA shared memory asynchronously.

    Requires NVIDIA compute capability 9.0 or newer and PTX 8.6 or newer.
    One thread issues the copy on behalf of the logical block. ``pointer`` and
    ``pred`` must be scalars. ``smem`` must be a one-dimensional, unswizzled
    shared-memory view; both addresses must be 16-byte aligned. This operation
    currently supports single-CTA kernels only.

    ``num_bytes`` is a positive compile-time multiple of 16, no larger than
    ``smem.nbytes_per_cta``. Only that prefix is copied. Compiler dependencies
    and ConSan conservatively cover the entire destination view, so use a
    sliced view when independently reusing other parts of an allocation.

    Initialize ``barrier`` and call ``mbarrier.expect(barrier, num_bytes)``
    before issuing the copy. Wait on its current phase before reading or
    reusing the destination, modifying the source, or invalidating the barrier.
    When predicating the copy, predicate the expectation consistently. The
    wait makes the copied bytes visible to ordinary shared-memory accesses.
    Global writes made through the generic proxy must also be made visible to
    the asynchronous proxy before the copy (see PTX ``fence.proxy.async``).
    GSan records the source read at issue time; it does not currently detect
    all source modifications that race with the pending asynchronous copy.

    Args:
        smem: Destination shared-memory descriptor.
        pointer: Scalar global-memory pointer.
        num_bytes: Compile-time byte count.
        barrier: Completion mbarrier.
        pred: Scalar predicate. Defaults to True.
    """
    num_bytes = _unwrap_if_constexpr(num_bytes)
    _check(isinstance(num_bytes, int), lambda: "num_bytes must be a compile-time integer")
    pointer = _semantic.to_tensor(pointer)
    pred = _semantic.to_tensor(pred)
    _check(pointer.type.is_ptr(), lambda: "pointer must be a scalar global-memory pointer")
    _check(pred.type.is_bool(), lambda: "pred must be a scalar boolean")
    _semantic.builder.create_async_bulk_copy_global_to_local(smem.handle, pointer.handle, num_bytes, barrier.handle,
                                                            pred.handle)
