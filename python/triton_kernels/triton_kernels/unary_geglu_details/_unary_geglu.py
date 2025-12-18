import triton
import triton.language as tl


@triton.jit(repr=lambda _: "_unary_geglu")
def _unary_geglu_fn(x, alpha):
    x = x.to(tl.float32)
    s = x / (1 + tl.exp(-alpha * x))
    return tl.fma(s, x, s)
