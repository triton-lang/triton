from ..ampere import async_copy
from . import mbarrier, tma
from ... import _core

__all__ = ["async_copy", "fence_async_shared", "mbarrier", "tma"]


@_core.builtin
def fence_async_shared(cluster=False, _semantic=None):
    cluster = _core._unwrap_if_constexpr(cluster)
    _semantic.builder.create_fence_async_shared(cluster)
