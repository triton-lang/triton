import triton

from triton_kernels.swiglu_details._swiglu import compute_swiglu


@triton.jit(repr=lambda _: "_unary_swiglu")
def _unary_swiglu_fn(x, alpha):
    return compute_swiglu(x, x, 1.0, alpha, None)
