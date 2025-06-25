from ..ampere import async_copy
from . import mbarrier, tma
from ... import _core

__all__ = ["async_copy", "fence_async_shared", "mbarrier", "tma", "warpgroup_mma"]


@_core.builtin
def fence_async_shared(cluster=False, _semantic=None):
    cluster = _core._unwrap_if_constexpr(cluster)
    _semantic.builder.create_fence_async_shared(cluster)


@_core.builtin
def warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=0, is_async=False, _semantic=None):
    use_acc = _semantic.to_tensor(use_acc)

    if precision is None:
        precision = _semantic.builder.options.default_dot_input_precision

    precision = _semantic._str_to_dot_input_precision(precision)
    max_num_imprecise_acc = _core._unwrap_if_constexpr(max_num_imprecise_acc)
    is_async = _core._unwrap_if_constexpr(is_async)

    handle = _semantic.builder.create_warpgroup_mma(a.handle, b.handle, acc.handle, use_acc.handle, precision,
                                                    max_num_imprecise_acc, is_async)
    return _core.tensor(handle, acc.type)
