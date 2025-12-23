import math

import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon._runtime import constexpr_function, jit
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["allocate_mbarrier", "arrive", "init", "invalidate", "MBarrierLayout", "wait"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization in Ampere and later architectures.

    Args:
        cga_layout (List[List[int]]): CGA layout bases. Defaults to [].
    """

    def __init__(self, cga_layout=None):
        super().__init__(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=cga_layout or [])

    @staticmethod
    @constexpr_function
    def multicta(num_ctas: int, two_cta: bool = False):
        """
        Create a multi-CTA mbarrier layout.

        Args:
            num_ctas (int): Number of CTAs.
            two_cta (bool): Whether the barrier should synchronize every other CTA
        """
        num_ctas = ttgl._unwrap_if_constexpr(num_ctas)
        two_cta = ttgl._unwrap_if_constexpr(two_cta)
        if two_cta:
            assert num_ctas % 2 == 0, "num_ctas must be even for two-CTA mode"
        assert num_ctas > 0, "num_ctas must be positive"
        assert (num_ctas & (num_ctas - 1)) == 0, "num_ctas must be a power of two"

        bases = []
        if two_cta:
            bases.append([0])
            num_ctas //= 2

        for i in range(int(math.log2(num_ctas))):
            bases.append([2**i])
        return MBarrierLayout(bases)


@jit
def allocate_mbarrier(batch: ttgl.constexpr = None, two_ctas: ttgl.constexpr = False):
    """
    Helper function to allocate an mbarrier

    Args:
        two_ctas (bool): Whether the barrier should synchronize every other CTA
    """
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_elems: ttgl.constexpr = num_ctas if not two_ctas else num_ctas // 2
    ttgl.static_assert(batch is None or isinstance(batch.value, int))
    shape: ttgl.constexpr = [num_elems] if batch is None else [batch, num_elems]
    bar = ttgl.allocate_shared_memory(
        ttgl.int64,
        shape,
        MBarrierLayout.multicta(num_ctas=num_ctas, two_cta=two_ctas),
    )
    return bar


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
