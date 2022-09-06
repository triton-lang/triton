import torch

import triton
import triton.language as tl
import triton.runtime as runtime

# trigger the torch.device implicitly to ensure cuda context initialization
torch.zeros([10], device=torch.device('cuda'))


@triton.jit
def empty_kernel(X, stride_xm, BLOCK: tl.constexpr):
    pass


def test_empty_kernel_cubin_compile():

    device = torch.cuda.current_device()
    cubin = triton.compile(empty_kernel,
                           "*fp32,i32,i32",
                           device=device,
                           constants={"BLOCK": 256},
                           output="cubin")

    print('cubin size:', len(cubin))
    assert len(cubin) > 0


def test_empty_kernel_launch():
    device = torch.cuda.current_device()
    binary = runtime.build_kernel(empty_kernel, "*fp32,i32,i32",
                                  constants={"BLOCK": 256},
                                  num_warps=4,
                                  num_stages=3)
    grid = lambda META: (
        triton.cdiv(1024, META['BLOCK']) * triton.cdiv(1024, META['BLOCK']),
    )

    A = torch.zeros([1024], device="cuda")
    runtime.launch_kernel(kernel=binary,
                          grid=grid,
                          device=device,
                          X=A,
                          stride_xm=256,
                          BLOCK=tl.constexpr(256))
