from ..._semantic import _check
from ..._core import _unwrap_if_constexpr, builtin
from triton._C.libtriton import ir

__all__ = [
    "async_copy_global_to_shared",
    "mbarrier_arrive",
    "commit_group",
    "wait_group",
]


@builtin
def async_copy_global_to_shared(smem, pointer, mask=None, cache_modifier="", eviction_policy="", volatile=False,
                                _semantic=None):
    mask = _unwrap_if_constexpr(mask)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask_handle = mask.handle if mask is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, cache_modifier,
                                                        eviction_policy, volatile)


@builtin
def mbarrier_arrive(mbarrier, increment_count=True, _semantic=None):
    """Arrive on the mbarrier once all outstanding async copies are complete.
    """
    increment_count = _unwrap_if_constexpr(increment_count)
    _semantic.builder.create_async_copy_mbarrier_arrive(mbarrier.handle, increment_count)


@builtin
def commit_group(_semantic=None):
    _semantic.builder.create_async_commit_group()


@builtin
def wait_group(num_outstanding=0, _semantic=None):
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)
