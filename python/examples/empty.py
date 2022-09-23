import triton
import triton.language as tl
import torch


@triton.jit
def kernel(X, stride_xm, stride_xn, BLOCK: tl.constexpr):
    pass


X = torch.randn(1, device="cuda")
kernel[(1,)](X, 1, 1, BLOCK=1024)