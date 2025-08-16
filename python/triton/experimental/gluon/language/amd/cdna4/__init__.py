from triton.experimental.gluon.language import _core as ttgl
from ..._core import ir, builtin, float32, _unwrap_if_constexpr, shared_memory_descriptor
from ..._semantic import _check
from ..._layouts import DotOperandLayout, BlockedLayout, SliceLayout
from .._layouts import AMDMFMALayout
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from ..cdna3 import _verify_buffer_load_store

__all__ = [
    *__cdna3_all,
    "global_load_to_shared",
    "buffer_load_to_shared",
    "async_wait",
    "relax_shared",
    "mfma_scaled",
]


@builtin
def global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD Global load to shared operation. This operation loads data directly
    from global memory to shared memory without going through registers. This
    operation happens asynchronously and requires a subsequent `async_wait`
    to ensure the data is available in shared memory.

    While using this operation, the following conditions must be met or
    lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be at least 128 and access must be coalesced.
    - If `dest` is swizzled, it only can be swizzled across warp boundaries.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (tensor pointer): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        ptr, mask = _semantic.broadcast_impl_value(ptr, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        ptr, other = _semantic.broadcast_impl_value(ptr, other)

    _check(isinstance(ptr.type.layout, (BlockedLayout, SliceLayout)),
           lambda: "expected ptr type layout to be BlockedLayout or SliceLayout")
    _check(
        dest.shape == ptr.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {ptr.shape}")

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(dest.handle, ptr.handle, mask_handle, other_handle,
                                                        cache_modifier, ir.EVICTION_POLICY.NORMAL, False)


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD Buffer load to shared operation. Buffer load is similar to global load
    but it accesses global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation loads data directly
    from global memory to shared memory without going through registers. This
    operation happens asynchronously and requires a subsequent `async_wait`
    to ensure the data is available in shared memory.

    While using this operation, the following conditions must be met or
    lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be at least 128 and access must be coalesced.
    - If `dest` is swizzled, it only can be swizzled across warp boundaries.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    _check(isinstance(offsets.type.layout, (BlockedLayout, SliceLayout)),
           lambda: "expected offsets type layout to be BlockedLayout or SliceLayout")
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


@builtin
def mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled MFMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
    Currently supported only on CDNA4 hardware.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (tensor): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (tensor): Scale factor for operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b_format (str): Format of the operand B.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    assert a_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format.value}"
    assert b_format.value in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format.value}"

    tensor = _semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, False, True, True, float32)

    ret_ty = ttgl.distributed_type(tensor.dtype, tensor.shape, layout)
    return ttgl.tensor(tensor.handle, ret_ty)


class relaxed_shared_memory_descriptor(shared_memory_descriptor):
    """
    A wrapper for shared memory descriptor that relaxes the shared memory load
    to remove fence before the load operation.
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

        print('[DEBUG] Emitting relaxed load')
        return ret
