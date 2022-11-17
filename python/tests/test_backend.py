import triton
import triton.language as tl
import torch
import pytest
from .test_core import  numpy_random, to_triton

@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dtype", ['float16'])
def test_convert2d(dtype, shape, device='cuda'):
    
    src = f"#triton_gpu.blocked<{{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}}>"
    # dst = f"#triton_gpu.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}}>"
    dst = f"#triton_gpu.mma<{{version=2, warpsPerCTA=[1,4]}}>"

    ir = f"""
#src = {src}
#dst = {dst}
module attributes {{"triton_gpu.num-warps" = 4 : i32}} {{
  func public @convert(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: i32 {{tt.divisibility = 16 : i32}}, 
                       %arg2: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg3: i32 {{tt.divisibility = 16 : i32}}) {{
    %0 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #triton_gpu.slice<{{dim = 1, parent = #src}}>>
    %1 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #triton_gpu.slice<{{dim = 1, parent = #dst}}>>
    %2 = tt.splat %arg1 : (i32) -> tensor<64x1xi32, #src>
    %3 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #src>
    %4 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #triton_gpu.slice<{{dim = 0, parent = #src}}>>
    %5 = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #triton_gpu.slice<{{dim = 0, parent = #dst}}>>
    %6 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #dst>
    %7 = tt.splat %arg3 : (i32) -> tensor<1x64xi32, #dst>
    %8 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<64xi32, #triton_gpu.slice<{{dim = 1, parent = #src}}>>) -> tensor<64x1xi32, #src>
    %9 = arith.muli %8, %2 : tensor<64x1xi32, #src>
    %10 = tt.addptr %3, %9 : tensor<64x1x!tt.ptr<f16>, #src>
    %11 = tt.expand_dims %4 {{axis = 0 : i32}} : (tensor<64xi32, #triton_gpu.slice<{{dim = 0, parent = #src}}>>) -> tensor<1x64xi32, #src>
    %12 = tt.broadcast %11 : (tensor<1x64xi32, #src>) -> tensor<64x64xi32, #src>
    %13 = tt.broadcast %10 : (tensor<64x1x!tt.ptr<f16>, #src>) -> tensor<64x64x!tt.ptr<f16>, #src>
    %14 = tt.addptr %13, %12 : tensor<64x64x!tt.ptr<f16>, #src>
    %15 = tt.load %14 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<64x64xf16, #src>
    %16 = tt.expand_dims %1 {{axis = 1 : i32}} : (tensor<64xi32, #triton_gpu.slice<{{dim = 1, parent = #dst}}>>) -> tensor<64x1xi32, #dst>
    %17 = tt.addptr %6, %16 : tensor<64x1x!tt.ptr<f16>, #dst>
    %18 = tt.expand_dims %5 {{axis = 0 : i32}} : (tensor<64xi32, #triton_gpu.slice<{{dim = 0, parent = #dst}}>>) -> tensor<1x64xi32, #dst>
    %19 = arith.muli %18, %7 : tensor<1x64xi32, #dst>
    %20 = tt.broadcast %19 : (tensor<1x64xi32, #dst>) -> tensor<64x64xi32, #dst>
    %21 = tt.broadcast %17 : (tensor<64x1x!tt.ptr<f16>, #dst>) -> tensor<64x64x!tt.ptr<f16>, #dst>
    %22 = tt.addptr %21, %20 : tensor<64x64x!tt.ptr<f16>, #dst>
    %23 = triton_gpu.convert_layout %15 : (tensor<64x64xf16, #src>) -> tensor<64x64xf16, #dst>
    tt.store %22, %23 : tensor<64x64xf16, #dst>
    return
  }}
}}
"""
    x = to_triton(numpy_random(shape, dtype_str=dtype))
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    handle = kernel[(1,1,1)](z.data_ptr(), shape[1], x.data_ptr(), shape[1])

    assert torch.equal(z, x)

