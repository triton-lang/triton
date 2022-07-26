import triton
import triton.language as tl

@triton.jit
def kernel(X, stride_xm, stride_xn, BLOCK: tl.constexpr):
    pass

ret = triton.compile(kernel, "*fp32,i32,i32", constants={"BLOCK": 256}, output="ttgir")