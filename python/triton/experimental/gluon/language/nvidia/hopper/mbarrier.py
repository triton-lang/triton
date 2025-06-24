from ..ampere.mbarrier import MBarrierLayout, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = ["arrive", "expect", "init", "invalidate", "MBarrierLayout", "wait"]


@builtin
def expect(mbarrier, bytes, pred=True, _semantic=None):
    bytes = _unwrap_if_constexpr(bytes)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes, pred.handle)


@builtin
def arrive(mbarrier, *, count=1, pred=True, _semantic=None):
    count = _unwrap_if_constexpr(count)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle)
