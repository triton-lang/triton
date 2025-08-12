from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["arrive", "init", "invalidate", "MBarrierLayout", "wait"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization in Ampere and later architectures.

    Args:
        ctas_per_cga (int): CTAs per CGA grouping. Defaults to 1.
        cta_split_num (int): CTA split factor. Defaults to 1.
    """

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
    """
    Initialize an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The initial count for the barrier.
    """
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_mbarrier_init(mbarrier.handle, count)


@builtin
def invalidate(mbarrier, _semantic=None):
    """
    Invalidate an mbarrier, resetting its state.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to invalidate.
    """
    _semantic.builder.create_mbarrier_inval(mbarrier.handle)


@builtin
def wait(mbarrier, phase, pred=True, deps=(), _semantic=None):
    """
    Wait until the mbarrier object completes its current phase.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase index to wait for.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
        deps (Sequence[shared_memory_descriptor]): Dependent allocations barrier is waiting on. Used to track liveness of dependent allocations. Defaults to ().
    """
    phase = _semantic.to_tensor(phase)
    pred = _semantic.to_tensor(pred)
    deps = [x.handle for x in deps]
    _semantic.builder.create_mbarrier_wait(mbarrier.handle, phase.handle, pred.handle, deps)


@builtin
def arrive(mbarrier, *, pred=True, _semantic=None):
    """
    Arrive on an mbarrier, signaling that a thread has reached the barrier.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to arrive on.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = 1
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle)
