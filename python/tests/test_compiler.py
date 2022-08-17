import torch

import triton
import triton.language as tl

# trigger the torch.device implicitly to ensure cuda context initialization
torch.zeros([10], device=torch.device('cuda'))


def test_empty_kernel_cubin_compile():
    @triton.jit
    def kernel(X, stride_xm, stride_xn, BLOCK: tl.constexpr):
        pass

    device = torch.cuda.current_device()
    cubin = triton.compile(kernel,
                           "*fp32,i32,i32",
                           device=device,
                           constants={"BLOCK": 256},
                           output="cubin")

    print('cubin size:', len(cubin))
    assert len(cubin) > 0
