import torch
from torch.testing import assert_close

import triton
import triton.language as tl
import triton.runtime as runtime


def vecadd_no_scf_tester(num_warps, block_size, tensor_shape):
    @triton.jit
    def kernel(x_ptr,
               y_ptr,
               z_ptr,
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

    torch.zeros([10], device=torch.device('cuda'))
    device = torch.cuda.current_device()
    binary = runtime.build_kernel(kernel, "*fp32,*fp32,*fp32,i32",
                                  constants={"BLOCK_SIZE_N": block_size},
                                  num_warps=num_warps,
                                  num_stages=3)

    x = torch.randn(tensor_shape, device='cuda', dtype=torch.float32)
    y = torch.randn(tensor_shape, device='cuda', dtype=torch.float32)
    z = torch.empty(tensor_shape, device=x.device, dtype=x.dtype)

    grid = lambda EA: (triton.cdiv(x.shape.numel(), block_size),)

    runtime.launch_kernel(kernel=binary,
                          grid=grid,
                          device=device,
                          x_ptr=x,
                          y_ptr=y,
                          z_ptr=z,
                          BLOCK_SIZE_N=tl.constexpr(block_size))
    golden_z = x + y
    assert_close(z, golden_z, rtol=1e-7, atol=1e-7)


def test_vecadd_no_scf():
    vecadd_no_scf_tester(num_warps=2, block_size=256, tensor_shape=(256,))
    vecadd_no_scf_tester(num_warps=1, block_size=256, tensor_shape=(256,))

    # masked load/store
    vecadd_no_scf_tester(num_warps=1, block_size=256, tensor_shape=(301,))
    vecadd_no_scf_tester(num_warps=2, block_size=256, tensor_shape=(513,))


if __name__ == '__main__':
    test_vecadd_no_scf()
