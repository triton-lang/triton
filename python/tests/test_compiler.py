import torch

import triton
import triton.language as tl

# trigger the torch.device implicitly to ensure cuda context initialization
torch.zeros([10], device=torch.device('cuda'))


@triton.jit
def empty_kernel(X, stride_xm, BLOCK: tl.constexpr):
    pass


def test_empty_kernel_cubin_compile():

    device = torch.cuda.current_device()
    kernel = triton.compile(empty_kernel,
                            "*fp32,i32,i32",
                            device=device,
                            constants={"BLOCK": 256})

    assert len(kernel.asm["cubin"]) > 0


def test_empty_kernel_launch():
    grid = lambda META: (
        triton.cdiv(1024, META['BLOCK']) * triton.cdiv(1024, META['BLOCK']),
    )

    A = torch.zeros([1024], device="cuda")
    empty_kernel[grid](X=A, stride_xm=256, BLOCK=256)
