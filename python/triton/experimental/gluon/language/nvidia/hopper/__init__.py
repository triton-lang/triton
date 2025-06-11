from . import mbarrier
from . import tma
from ... import _core

__all__ = ["fence_async_shared", "mbarrier", "tma"]


@_core.builtin
def fence_async_shared(cluster=False, _semantic=None):
    cluster = _core._unwrap_if_constexpr(cluster)
    _semantic.builder.create_fence_async_shared(cluster)
