import math

import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon._runtime import constexpr_function, jit
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "init",
    "init_tcgen05_mma",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


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
def init(mbarrier, count, *, fallback_count=None, _semantic=None):
    """
    Initialize an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The initial count per CTA sharing the barrier.
        fallback_count (int, optional): Count per CTA when a preferred cluster
            falls back to two CTAs. Defaults to ``count``.

    Both counts are multiplied by the number of CTAs sharing the barrier.
    """
    count = _unwrap_if_constexpr(count)
    fallback_count = _unwrap_if_constexpr(fallback_count)
    _semantic.builder.create_mbarrier_init(mbarrier.handle, count, fallback_count)


@builtin
def init_tcgen05_mma(mbarrier, descs=(), *, two_ctas, _semantic=None):
    """
    Initialize a tcgen05 completion barrier for preferred and fallback clusters.

    Args:
        mbarrier (shared_memory_descriptor): A barrier with one element per CTA,
            allocated with ``allocate_mbarrier()``.
        descs (Sequence): Shared or tensor memory descriptors used for multicast
            completion, matching those passed to ``tcgen05_commit`` or the
            multicast MMA. Use an empty sequence without multicast.
        two_ctas (bool): Whether the MMA uses ``cta_group::2``. This must match
            the accumulator's ``two_ctas`` layout setting.

    For example::

        bar = mbarrier.allocate_mbarrier()
        mbarrier.init_tcgen05_mma(bar, [a, b], two_ctas=acc.layout.two_ctas)

    Descriptor layouts determine both counts: broadcasting across CTA bit 0
    remains within a fallback pair; higher broadcast bits are removed.
    """
    from ..blackwell import tcgen05_mma_barrier_count

    descs = _unwrap_if_constexpr(descs)
    two_ctas = _unwrap_if_constexpr(two_ctas)
    num_ctas = _unwrap_if_constexpr(ttgl.num_ctas(_semantic=_semantic))
    cga_layout = [[2**i] for i in range(num_ctas.bit_length() - 1)]
    actual_cga_layout = [list(basis) for basis in mbarrier.layout.cga_layout]
    ttgl.static_assert(
        list(mbarrier.shape) == [num_ctas] and actual_cga_layout == cga_layout,
        "tcgen05 MMA barriers must have shape [num_ctas] and 1D cga_layout",
        _semantic=_semantic,
    )
    count = tcgen05_mma_barrier_count(descs, multicast=bool(descs), two_ctas=two_ctas)
    fallback_count = tcgen05_mma_barrier_count(descs, multicast=bool(descs), two_ctas=two_ctas,
                                               cluster_size=min(num_ctas, 2))
    init(mbarrier, count, fallback_count=fallback_count, _semantic=_semantic)


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
def arrive(mbarrier, *, pred=True, from_cta=None, _semantic=None):
    """
    Arrive on an mbarrier, signaling that a thread has reached the barrier.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to arrive on.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
        from_cta (int, optional): Mask of CTA-ID bits preserved when routing the arrival, in
            ``[0, num_ctas - 1]``. Defaults to ``num_ctas - 1``, which arrives from each CTA to itself; ``0``
            routes from CTA 0 to every CTA.
    """
    count = 1
    multicast_cta = 0
    from_cta = _unwrap_if_constexpr(from_cta)
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle, from_cta, multicast_cta)
