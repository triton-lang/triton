import torch
import triton


@triton.jit
def kernel(X):
    pass


x = torch.empty([5], device="cuda")
kernel[(1, )](x)
