from ..ampere import async_copy
from . import mbarrier, tma
from ... import _core

__all__ = ["async_copy", "fence_async_shared", "mbarrier", "tma", "warpgroup_mma", "warpgroup_mma_wait"]


@_core.builtin
def fence_async_shared(cluster=False, _semantic=None):
    """
    Issue a fence to complete asynchronous shared memory operations.

    Args:
        cluster (bool): Whether to fence across cluster. Defaults to False.
    """
    cluster = _core._unwrap_if_constexpr(cluster)
    _semantic.builder.create_fence_async_shared(cluster)


@_core.builtin
def warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False,
                  _semantic=None):
    """
    Perform warpgroup MMA (Tensor Core) operations.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (tensor or shared_memory_descriptor): Left hand side operand.
        b (tensor or shared_memory_descriptor): Right hand side operand.
        acc (tensor): Accumulator tensor.
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        precision (str, optional): Dot input precision. Defaults to builder default.
        max_num_imprecise_acc (int): Max imprecise accumulations. Used for fp8 -> fp32 dot. Determines how many accumulation are done in limited precision. Defaults to None, which means no upcasting is done.
        is_async (bool): Whether operation is asynchronous. Defaults to False.

    Returns:
        tensor: Result of warpgroup MMA operation.
    """
    use_acc = _semantic.to_tensor(use_acc)

    if precision is None:
        precision = _semantic.builder.options.default_dot_input_precision

    precision = _semantic._str_to_dot_input_precision(precision)

    K = a.type.shape[-1]
    if max_num_imprecise_acc is None:
        if a.dtype.is_fp8() and b.dtype.is_fp8():
            max_num_imprecise_acc = _semantic.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        if a.dtype.is_fp8() and b.dtype.is_fp8() and max_num_imprecise_acc > K:
            raise ValueError(f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")

    max_num_imprecise_acc = _core._unwrap_if_constexpr(max_num_imprecise_acc)
    is_async = _core._unwrap_if_constexpr(is_async)

    handle = _semantic.builder.create_warpgroup_mma(a.handle, b.handle, acc.handle, use_acc.handle, precision,
                                                    max_num_imprecise_acc, is_async)
    return _core.tensor(handle, acc.type)


@_core.builtin
def warpgroup_mma_wait(num_outstanding=0, deps=None, _semantic=None):
    """
    Wait until `num_outstanding` or less warpgroup MMA operations are in-flight.

    Args:
        num_outstanding (int): Number of outstanding warpgroup MMA operations to wait for. Defaults to 0.
        deps (Sequence[tensor]): List of dependencies that need to be kept alive while the mma is unfinished.
    """
    deps = [x.handle for x in deps] if deps is not None else []
    num_outstanding = _core._unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_warpgroup_mma_wait(deps, num_outstanding)
