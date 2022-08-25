import torch
from torch.testing import assert_allclose

import triton
import triton.language as tl
import triton.runtime as runtime

NUM_WARPS = 4
BLOCK_SIZE = 256

# triton kernel


def test_vecadd_no_scf():
    @triton.jit
    def kernel(x_ptr, stride_xn,
               y_ptr, stride_yn,
               z_ptr, stride_zn,
               BLOCK_SIZE_N: tl.constexpr):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_ptrs = x_ptr + offset
        y_ptrs = y_ptr + offset
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        z = x + y
        z_ptrs = z_ptr + offset
        tl.store(z_ptrs, z)

    # TODO: add this to CI, to make sure the the compilation flow is at lease OK
    #       before we have GPU machines for CI.
    # ptx, shem_size, kernel_name = triton.compile(kernel,
    #                                              "*fp32,i32,*fp32,i32,*fp32,i32",
    #                                              constants={"BLOCK_SIZE_N": 256},
    #                                              num_warps=NUM_WARPS,
    #                                              device=0, output="ptx")

    torch.zeros([10], device=torch.device('cuda'))
    device = torch.cuda.current_device()
    binary = runtime.build_kernel(kernel, "*fp32,i32,*fp32,i32,*fp32,i32",
                                  device=device,
                                  constants={"BLOCK_SIZE_N": BLOCK_SIZE},
                                  num_warps=NUM_WARPS,
                                  num_stages=3)
    grid = lambda META: (1, )

    x = torch.randn((256,), device='cuda', dtype=torch.float32)
    y = torch.randn((256,), device='cuda', dtype=torch.float32)
    z = torch.empty((256,), device=x.device, dtype=x.dtype)
    runtime.launch_kernel(fn=kernel,
                          binary=binary,
                          grid=grid,
                          num_warps=NUM_WARPS,
                          num_stages=3,
                          x_ptr=x,
                          stride_xn=x.stride(0),
                          y_ptr=y,
                          stride_yn=y.stride(0),
                          z_ptr=z,
                          stride_zn=z.stride(0),
                          BLOCK_SIZE_N=tl.constexpr(BLOCK_SIZE))
    golden_z = x + y
    assert_allclose(z, golden_z, rtol=1e-7, atol=1e-7)
