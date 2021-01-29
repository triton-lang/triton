import torch
import triton
import os

kernels = dict()
def get_kernel(block, dtype, device):
    key = (block, dtype, device)
    if key not in kernels:
        src = triton.read(os.path.join(os.path.dirname(__file__), 'softmax.c'))
        defines = {'BLOCK': block, 'TYPE': dtype}
        kernels[key] = triton.kernel(src, device = device, defines = defines)
    return kernels[key]


class _softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = torch.empty_like(x)
        M, N = x.shape
        kernel = get_kernel(N, x.dtype, x.device)
        kernel(x.data_ptr(), y.data_ptr(), grid = lambda opt: [M, ])
        return y

softmax = _softmax.apply
        

