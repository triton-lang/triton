"""
Asynchronous global-to-shared memory copies for CDNA4.

These load data from global memory directly into shared memory without going through registers, either through a
tensor of pointers (:func:`global_load_to_shared`) or a scalar base pointer and a tensor of offsets
(:func:`buffer_load_to_shared`, deprecated in favor of
:func:`buffer.load_to_shared <triton.experimental.gluon.language.amd.cdna4.buffer.load_to_shared>`). Group issued
copies with :func:`commit_group` and wait for them with :func:`wait_group` before reading the destination.
"""

import warnings

from triton.experimental.gluon.language import _core as ttgl
from ..._core import ir, builtin, _unwrap_if_constexpr
from ..._semantic import _check
from ..._layouts import DistributedLayout
from ..cdna3 import CachePolicy, _verify_buffer_ops

__all__ = [
    "global_load_to_shared",
    "buffer_load_to_shared",
    "commit_group",
    "wait_group",
    "load_shared_relaxed",
]


def _cache_policy_attr(cache_policy, cache_modifier, semantic):
    """Return the `cachePolicy` attribute of a copy from a gfx9 `cache_policy` or a `cache_modifier` string."""
    cache_policy = _unwrap_if_constexpr(cache_policy)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    if cache_policy is None:
        return ttgl.CachePolicy(cache_modifier or "none")._to_ir(semantic.builder)
    _check(not cache_modifier, lambda: "cache_policy and cache_modifier are mutually exclusive")
    _check(isinstance(cache_policy, CachePolicy),
           lambda: f"cache_policy must be a {CachePolicy.__qualname__}, got {cache_policy!r}")
    return cache_policy._to_ir(semantic.builder)


@builtin
def global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="", *, cache_policy=None, _semantic=None):
    """
    AMD global load to shared operation.

    This operation loads data directly from global memory to shared memory
    without going through registers. It happens asynchronously; call
    :func:`wait_group` before accessing ``dest``. The operation still completes
    in order with :func:`~triton.experimental.gluon.language.load`,
    :func:`~triton.experimental.gluon.language.store`,
    :func:`~triton.experimental.gluon.language.amd.cdna4.buffer_load`, and
    :func:`~triton.experimental.gluon.language.amd.cdna4.buffer_store` on
    CDNA4, so interleaving with them will hurt performance.

    Compared to :func:`buffer_load_to_shared`, it requires a tensor pointer which
    supports 64-bit indexing range for each thread in a block, which gives more
    flexibility, but at the cost of higher register pressure and no hardware
    out-of-bound masking support. Prefer :func:`buffer_load_to_shared` when
    possible for better performance.

    When ``mask`` is provided without ``other``, masked elements in ``dest``
    are left unchanged. Providing ``other`` writes that value to masked
    elements instead.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    * For the ``ptr`` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    * Writes to ``dest`` must be coalesced.
    * If ``dest`` is swizzled, it can only be swizzled within a warp boundary.
    * If a ``mask`` is present it must be aligned to the load's vector width: each
      group of (size per thread) consecutive mask values must be identical. The
      copy transfers each lane's whole vector in a single transaction, so a mask
      whose true/false boundary cannot be proven vector-aligned forces
      per-element vectorization and fails to lower.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer tensor): Tensor of pointers to global memory to load from.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing values for masked elements. When omitted, masked
            elements in ``dest`` are left unchanged. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
        cache_policy (CachePolicy, optional): gfx9 cache hints. Cannot be combined with ``cache_modifier``.
            Defaults to None.
    """
    _check(ptr.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(ptr.type.layout, DistributedLayout), lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        dest.shape == ptr.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {ptr.shape}")

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        ptr, mask = _semantic.broadcast_impl_value(ptr, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.dtype.element_ty)
        ptr, other = _semantic.broadcast_impl_value(ptr, other)

    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    cache_policy = _cache_policy_attr(cache_policy, cache_modifier, _semantic)
    _semantic.builder.create_async_copy_global_to_local(dest.handle, ptr.handle, mask_handle, other_handle,
                                                        cache_policy, False)


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    Deprecated. Use :func:`buffer.load_to_shared <triton.experimental.gluon.language.amd.cdna4.buffer.load_to_shared>`
    with a descriptor from :func:`buffer.make_buffer_descriptor <triton.experimental.gluon.language.amd.cdna3.buffer.make_buffer_descriptor>` instead.

    AMD buffer load to shared operation.

    Buffer load is similar to :func:`global_load_to_shared`, but it accesses
    global memory through a scalar base pointer and a tensor of 32-bit offsets
    rather than a tensor of pointers. This operation loads data directly from
    global memory to shared memory without going through registers. It happens
    asynchronously; call :func:`wait_group` before accessing ``dest``. The
    operation still completes in order with
    :func:`~triton.experimental.gluon.language.load`,
    :func:`~triton.experimental.gluon.language.store`,
    :func:`~triton.experimental.gluon.language.amd.cdna4.buffer_load`, and
    :func:`~triton.experimental.gluon.language.amd.cdna4.buffer_store` on
    CDNA4, so interleaving with them will hurt performance.

    Compared to :func:`global_load_to_shared`, it has better performance and also
    supports hardware out-of-bound masking. But it strictly requires a
    32-bit offset instead of a 64-bit tensor pointer.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    * For the ``offsets`` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    * Writes to ``dest`` must be coalesced.
    * If ``dest`` is swizzled, it can only be swizzled within a warp boundary.
    * If a ``mask`` is present it must be aligned to the load's vector width: each
      group of (size per thread) consecutive mask values must be identical. The
      copy transfers each lane's whole vector in a single transaction, so a mask
      whose true/false boundary cannot be proven vector-aligned forces
      per-element vectorization and fails to lower.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Requires ``other`` when specified. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing values for masked elements. Buffer loads cannot
            preserve masked destination elements, so this is required when ``mask`` is specified. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(isinstance(offsets.type.layout, DistributedLayout),
           lambda: "expected offsets type layout to be a DistributedLayout")
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    _check(
        mask is None or other is not None, lambda:
        "other is required for a masked buffer_load_to_shared because masked buffer loads cannot preserve destination shared memory"
    )
    _verify_buffer_ops(ptr, offsets, mask, other)

    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.type.scalar.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


@builtin
def commit_group(_semantic=None):
    """
    Commit outstanding async operations.

    This finalizes a set of async copy operations which can be waited upon via :func:`wait_group`.
    """
    _semantic.builder.create_async_commit_group()


@builtin
def wait_group(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding commit groups.

    It will block until the number of outstanding commit groups is less than or equal to
    ``num_outstanding``. Note that uncommitted async operations will be waited upon even if
    ``num_outstanding`` is 0.

    Args:
        num_outstanding (int): The number of outstanding commit groups to wait for. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


@builtin
def load_shared_relaxed(smem, layout, _semantic=None):
    """
    Deprecated. Use the shared memory descriptor's load method instead,
    for example `smem.load(layout)`.

    This compatibility wrapper emits a normal shared-memory load.

    Args:
        smem (shared_memory_descriptor): Shared memory descriptor to load from.
        layout (DistributedLayout): The destination layout of the tensor.

    Returns:
        tensor: A Gluon tensor containing the loaded data.
    """
    warnings.warn(
        "cdna4.async_copy.load_shared_relaxed is deprecated,"
        " and will be removed in a future release."
        " Please use the shared memory descriptor's load method instead,"
        " for example smem.load(layout).",
        DeprecationWarning,
        stacklevel=3,
    )

    layout = _unwrap_if_constexpr(layout)
    return _semantic.shared_load(smem, layout)
