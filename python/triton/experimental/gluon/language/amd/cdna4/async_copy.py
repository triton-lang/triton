from ..._core import ir, builtin, _unwrap_if_constexpr, shared_memory_descriptor
from ..._semantic import _check
from ..._layouts import BlockedLayout, SliceLayout
from ..cdna3 import _verify_buffer_load_store

__all__ = [
    "global_load_to_shared",
    "buffer_load_to_shared",
    "async_wait",
    "relax_shared",
]


@builtin
def global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD global load to shared operation. This operation loads data directly
    from global memory to shared memory without going through registers. It
    operation happens asynchronously and requires a subsequent `async_wait`
    to ensure the data is available in shared memory. Compared to
    `buffer_load_to_shared`, it requires a tensor pointer which supports
    64-bit indexing range for each thread in a block, which gives more
    flexibility, but at the cost of higher register pressure and no hardware
    out-of-bound masking support. Prefer to use `buffer_load_to_shared` when
    possible for better performance.

    While using this operation, the following conditions must be met or
    lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be 128 or 32.
    - If `dest` is swizzled, it only can be swizzled within warp boundary.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer tensor): Tensor of pointers to global memory to load from.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(ptr.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(ptr.type.layout, (BlockedLayout, SliceLayout)),
           lambda: "expected ptr type layout to be BlockedLayout or SliceLayout")
    _check(
        dest.shape == ptr.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {ptr.shape}")

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        ptr, mask = _semantic.broadcast_impl_value(ptr, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        ptr, other = _semantic.broadcast_impl_value(ptr, other)

    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(dest.handle, ptr.handle, mask_handle, other_handle,
                                                        cache_modifier, ir.EVICTION_POLICY.NORMAL, False)


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD buffer load to shared operation. Buffer load is similar to global load
    but it accesses global memory via a scalar base pointer and a tensor of
    32-bit offsets instead of a tensor of pointers. This operation loads data
    directly from global memory to shared memory without going through
    registers. It happens asynchronously and requires a subsequent `async_wait`
    to ensure the data is available in shared memory.

    While using this operation, the following conditions must be met or
    lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be 128 or 32.
    - If `dest` is swizzled, it only can be swizzled within warp boundary.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(isinstance(offsets.type.layout, (BlockedLayout, SliceLayout)),
           lambda: "expected offsets type layout to be BlockedLayout or SliceLayout")

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    _verify_buffer_load_store(ptr, offsets, mask, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


@builtin
def async_wait(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding asynchronous memory operations, including
    `global_load_to_shared` and `buffer_load_to_shared`. It will block until
    the number of outstanding operations is less than or equal to `num_outstanding`.

    Args:
        num_outstanding (int): The number of outstanding operations to wait for. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


@builtin
def relax_shared(smem, _semantic=None):
    """
    Turn a shared memory descriptor into a relaxed shared memory descriptor.

    Args:
        smem (shared_memory_descriptor): The shared memory descriptor to relax.
    """
    return relaxed_shared_memory_descriptor(smem)


class relaxed_shared_memory_descriptor(shared_memory_descriptor):
    """
    A wrapper for shared memory descriptor that relaxes the shared memory load
    to give hints to underlying compiler to better optimize and avoid
    unnecessary waits
    """
    SYNCED_VIA_WAIT_ATTR_NAME = "ttg.amdgpu.syncedViaAsyncWait"

    def __init__(self, smem):
        self.handle = smem.handle
        self.type = smem.type

    @builtin
    def load(self, layout, _semantic):
        """
        Load a tensor from shared memory.

        Args:
            layout (DistributedLayout): The destination layout of the tensor.

        Returns:
            tensor: A Gluon tensor containing the loaded data.
        """
        layout = _unwrap_if_constexpr(layout)
        ret = _semantic.shared_load(self, layout)
        ret.handle.set_attr(self.SYNCED_VIA_WAIT_ATTR_NAME, _semantic.builder.get_bool_attr(True))
        return ret
