from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["MBarrierLayout", "init", "invalidate", "expect", "wait", "arrive"]


class MBarrierLayout(SwizzledSharedLayout):

    def __init__(self, ctas_per_cga: int = 1, cta_split_num: int = 1):
        super().__init__(
            vec=1,
            per_phase=1,
            max_phase=1,
            order=[0],
            ctas_per_cga=[ctas_per_cga],
            cta_split_num=[cta_split_num],
            cta_order=[0],
        )


@builtin
def init(mbarrier, count, _builder=None):
    count = _unwrap_if_constexpr(count)
    _builder.create_mbarrier_init(mbarrier.handle, count)


@builtin
def invalidate(mbarrier, _builder=None):
    _builder.create_mbarrier_inval(mbarrier.handle)


@builtin
def expect(mbarrier, bytes, pred=True, _builder=None):
    bytes = _unwrap_if_constexpr(bytes)
    pred = ttgl.to_tensor(pred, _builder=_builder)
    _builder.create_mbarrier_expect(mbarrier.handle, bytes, pred.handle)


@builtin
def wait(mbarrier, phase, pred=True, deps=(), _builder=None):
    phase = ttgl.to_tensor(phase, _builder=_builder)
    pred = ttgl.to_tensor(pred, _builder=_builder)
    deps = [x.handle for x in deps]
    _builder.create_mbarrier_wait(mbarrier.handle, phase.handle, pred.handle, deps)


@builtin
def arrive(mbarrier, count, pred=True, _builder=None):
    count = _unwrap_if_constexpr(count)
    pred = ttgl.to_tensor(pred, _builder=_builder)
    _builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle)
