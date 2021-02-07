import torch
import triton
import os

fwd_src = triton.read(os.path.join(os.path.dirname(__file__), 'softmax.c'), kernel_names=['forward'])
fwd_kernels = dict()

def get_fwd_kernel(block, dtype, device):
    key = (block, dtype, device)
    if key not in fwd_kernels:
        defines = {'BLOCK': block, 'TYPE': dtype}
        fwd_kernels[key] = triton.kernel(fwd_src, device=device, defines=defines)
    return fwd_kernels[key]

class _softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.empty_like(x)
        M, N = x.shape
        kernel = get_fwd_kernel(N, x.dtype, x.device)
        grid = lambda opt: (M, )
        kernel(x.data_ptr(), y.data_ptr(), grid=grid)
        return y

softmax = _softmax.apply
