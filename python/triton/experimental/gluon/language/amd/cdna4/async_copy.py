from ..._core import ir, builtin, _unwrap_if_constexpr
from ..._semantic import _check
from ..._layouts import BlockedLayout, SliceLayout
from ..cdna3 import _verify_buffer_ops

__all__ = [
    "global_load_to_shared",
    "buffer_load_to_shared",
    "async_wait",
    "load_shared_relaxed",
]


@builtin
def global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD global load to shared operation. This operation loads data directly
    from global memory to shared memory without going through registers. It
    happens asynchronously and requires a subsequent `async_wait` to ensure the
    data is available in shared memory.
    Compared to `buffer_load_to_shared`, it requires a tensor pointer which
    supports 64-bit indexing range for each thread in a block, which gives more
    flexibility, but at the cost of higher register pressure and no hardware
    out-of-bound masking support. Prefer to use `buffer_load_to_shared` when
    possible for better performance.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    - Writes to `dest` must be coalesced.
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
    Compared to `global_load_to_shared`, it has better performance and also
    supports hardware out-of-bound masking. But it strictly requires a
    32-bit offset instead of a 64-bit tensor pointer.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    - For the `offsets` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    - Writes to `dest` must be coalesced.
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
    _verify_buffer_ops(ptr, offsets, mask, other)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


@builtin
def async_wait(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding memory operations, this includes normal load like
    `load` and `buffer_load`, as well as direct load to shared memory
    like `global_load_to_shared` and `buffer_load_to_shared`.
    It will block until the number of outstanding memory operations is less than
    or equal to `num_outstanding`.

    Args:
        num_outstanding (int): The number of outstanding operations to wait for. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


@builtin
def load_shared_relaxed(smem, layout, _semantic=None):
    """
    Load a tensor from shared memory with extra hints for the underlying
    compiler to avoid emitting unnecessary waits before loading from the target
    shared memory.

    Args:
        smem (shared_memory_descriptor): Shared memory descriptor to load from.
        layout (DistributedLayout): The destination layout of the tensor.

    Returns:
        tensor: A Gluon tensor containing the loaded data.
    """
    SYNCED_VIA_WAIT_ATTR_NAME = "ttg.amdgpu.syncedViaAsyncWait"

    layout = _unwrap_if_constexpr(layout)
    ret = _semantic.shared_load(smem, layout)
    ret.handle.set_attr(SYNCED_VIA_WAIT_ATTR_NAME, _semantic.builder.get_bool_attr(True))
    return ret
