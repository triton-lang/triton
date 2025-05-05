import triton
import triton.language as tl


@tl.constexpr_function
def log2(n):
    return len(bin(n)) - 3


@tl.constexpr_function
def _permute_to_end_order(n: int, axis: int):
    """
    Returns the order of the axes of a tensor to permute `axis` to the end.
    """
    order = tuple(range(n))
    return order[:axis] + order[(axis + 1):] + (axis, )


@triton.jit
def permute_to_end(x, axis: tl.constexpr):
    """
    Permutes `x` so that `axis` is the last axis.
    """
    N: tl.constexpr = len(x.shape)
    return tl.permute(x, _permute_to_end_order(N, axis).value)


@triton.jit
def split_n(x, N: tl.constexpr):
    """
    Given `x`, a tensor of shape AxB...x2x2...x2, split it N times.
    Return a tuple of the results.
    """
    xs = (x, )
    for i in tl.static_range(N):
        next = tl.split(xs[0])
        for j in tl.static_range(2**i - 1):
            next = next + tl.split(xs[j + 1])
        xs = next
    return xs
