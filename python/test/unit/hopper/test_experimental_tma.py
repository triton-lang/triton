import numpy as np
import pytest
import torch
import tempfile

import triton
import triton.language as tl


def test_descriptor_load_ttgir():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    SIZE = 128

    ir = f"""
    #blocked = #triton_gpu.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
    #shared = #triton_gpu.shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}}>
    module attributes {{"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
      tt.func public @kernel(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i8> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
        %c0_i32 = arith.constant 0 : i32
        %0 = tt.make_range {{end = {SIZE} : i32, start = 0 : i32}} : tensor<{SIZE}xi32, #blocked>
        %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<{SIZE}xf32, #shared, mutable>
        %2 = triton_gpu.local_alloc  : () -> !tt.memdesc<1xi64, #shared, mutable>
        triton_nvidia_gpu.init_barrier %2, 1 : <1xi64, #shared, mutable>
        triton_nvidia_gpu.async_tma_copy_global_to_local %arg1[%c0_i32] %1, %2 : <i8>, <1xi64, #shared, mutable> -> <{SIZE}xf32, #shared, mutable>
        triton_nvidia_gpu.wait_barrier %2, %c0_i32 : <1xi64, #shared, mutable>
        %3 = triton_gpu.local_load %1 : !tt.memdesc<{SIZE}xf32, #shared, mutable> -> tensor<{SIZE}xf32, #blocked>
        %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{SIZE}x!tt.ptr<f32>, #blocked>
        %5 = tt.addptr %4, %0 : tensor<{SIZE}x!tt.ptr<f32>, #blocked>, tensor<{SIZE}xi32, #blocked>
        tt.store %5, %3 : tensor<{SIZE}x!tt.ptr<f32>, #blocked>
        tt.return
      }}
    }}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, 4, desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)
    kernel[(1, 1, 1)](z_tri, desc)
    assert torch.equal(x, z_tri)


def test_experimetal_descriptor_load():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Test requires Hopper target.")
        return
    device = "cuda"
    SIZE = 128

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr):
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype)
        tl.store(Z + off, x)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    desc = np.empty(SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, 4, desc)
    desc = torch.tensor(desc, device=device)
    z_tri = torch.empty_like(x)
    kernel[(1, )](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri)
