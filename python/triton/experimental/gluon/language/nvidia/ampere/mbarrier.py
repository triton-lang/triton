from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
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
def init(mbarrier, count, _semantic=None):
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_mbarrier_init(mbarrier.handle, count)


@builtin
def invalidate(mbarrier, _semantic=None):
    _semantic.builder.create_mbarrier_inval(mbarrier.handle)


@builtin
def expect(mbarrier, bytes, pred=True, _semantic=None):
    bytes = _unwrap_if_constexpr(bytes)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes, pred.handle)


@builtin
def wait(mbarrier, phase, pred=True, deps=(), _semantic=None):
    phase = _semantic.to_tensor(phase)
    pred = _semantic.to_tensor(pred)
    deps = [x.handle for x in deps]
    _semantic.builder.create_mbarrier_wait(mbarrier.handle, phase.handle, pred.handle, deps)


@builtin
def arrive(mbarrier, count, pred=True, _semantic=None):
    count = _unwrap_if_constexpr(count)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle)
