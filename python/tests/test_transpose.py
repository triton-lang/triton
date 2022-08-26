import pytest
import torch
from torch.testing import assert_allclose

import triton
import triton.language as tl
import triton.runtime as runtime


@triton.jit
def kernel(x_ptr, stride_xm,
           z_ptr, stride_zn,
           SIZE_M: tl.constexpr, SIZE_N: tl.constexpr):
    off_m = tl.arange(0, SIZE_M)
    off_n = tl.arange(0, SIZE_N)
    Xs = x_ptr + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = z_ptr + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))

# These sizes cover the case of:
# - blocked layout and sliced layout with block parent
#  -- blocked layout in which sizePerThread/threadsPerWarp/warpsPerCTA
#     need/need not to be wrapped
#  -- sliced layout incase sizePerThread need to be wrapped
#  -- different orders
# - LayoutConversion from blocked -> blocked
# - tt.Broadcast which requires for broadcast in either/both of
#   CTA/perThread level

# What is not covered and requires for TODO:
# - vectorization load/store of shared memory
# - multiple replication of layout conversion


@pytest.mark.parametrize('NUM_WARPS,SIZE_M,SIZE_N', [
    [1, 16, 16],
    [1, 32, 32],
    [1, 32, 64],
    [2, 64, 128],
    [2, 128, 64]
])
def test_convert_layout_impl(NUM_WARPS, SIZE_M, SIZE_N):
    # TODO: this is to initialize the cuda context since it is not properly
    #       dealed with in the existing runtime, remove this when the runtime
    #       is updated
    torch.zeros([10], device=torch.device('cuda'))
    device = torch.cuda.current_device()
    binary = runtime.build_kernel(kernel,
                                  "*fp32,i32,*fp32,i32",
                                  constants={"SIZE_M": SIZE_M,
                                             "SIZE_N": SIZE_N},
                                  num_warps=NUM_WARPS,
                                  num_stages=3)
    grid = lambda META: (1, )

    x = torch.randn((SIZE_M, SIZE_N), device='cuda', dtype=torch.float32)
    z = torch.empty((SIZE_N, SIZE_M), device=x.device, dtype=x.dtype)
    runtime.launch_kernel(kernel=binary,
                          device=device,
                          grid=grid,
                          x_ptr=x,
                          stride_xm=x.stride(0),
                          z_ptr=z,
                          stride_zn=z.stride(0),
                          SIZE_M=tl.constexpr(SIZE_M),
                          SIZE_N=tl.constexpr(SIZE_N))
    golden_z = torch.t(x)
    assert_allclose(z, golden_z, rtol=1e-7, atol=1e-7)
