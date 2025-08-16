from triton.experimental.gluon.language import _core as ttgl
from ..._core import ir, builtin, float32, _unwrap_if_constexpr
from ..._semantic import _check
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from ..cdna3 import _verify_buffer_load_store

__all__ = [
    *__cdna3_all,
    "async_copy_global_to_shared",
    "async_wait",
    "buffer_load_to_shared",
    "mfma_scaled",
]


@builtin
def async_copy_global_to_shared(dest, pointer, mask=None, other=None, cache_modifier="", eviction_policy="",
                                volatile=False, _semantic=None):
    """
    Asynchronously copy elements from global memory to shared memory.

    Limitations:

    - Only supports 2D tensors.
    - Require load bitwidth to be 128.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    _check(
        dest.shape == pointer.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {pointer.shape}"
    )
    _check(len(dest.shape) == 2, lambda: f"expected dest shape to be 2d but got {len(dest.shape)}d")

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(dest.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_modifier, ir.EVICTION_POLICY.NORMAL, False)


@builtin
def async_wait(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding asynchronous copy operations.

    Args:
        num_outstanding (int): Wait until `num_outstanding` or less async copy operations in-flight. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


@builtin
def synced_via_wait(v, _semantic=None):
    """
    Annotate a local load operation as synced via async wait, so the LLVM
    will not emit conservative wait counts

    Args:
        v (tensor): The tensor loaded from shared memory.
    """
    v.handle.set_attr("ttg.amdgpu.syncedViaAsyncWait", _semantic.builder.get_bool_attr(True))


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD Buffer load to shared operation. Buffer load is similar to normal load
    but it accesses global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into shared memory instead of registers.

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

    _verify_buffer_load_store(ptr, offsets, mask, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


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
