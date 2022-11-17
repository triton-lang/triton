import triton
import triton.language as tl
import torch
import pytest
from .test_core import  numpy_random, to_triton

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", ['float16'])
def test_convert2d(dtype, shape, device='cuda'):
    
    src_layout = "#triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>"
    dst_layout = "#triton_gpu.mma<{version = 2, warpsPerCTA = [1, 4]}>"

    ir = f"""
#src = {src_layout}
#dst = {dst_layout}
"""  + """
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func public @kernel_0d1d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<128> : tensor<128x1xi32, #src>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>
    %2 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #src>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #src}>>) -> tensor<128x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<128x1xi32, #src>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #src}>>) -> tensor<1x128xi32, #src>
    %7 = tt.broadcast %6 : (tensor<1x128xi32, #src>) -> tensor<128x128xi32, #src>
    %8 = tt.broadcast %5 : (tensor<128x1xi32, #src>) -> tensor<128x128xi32, #src>
    %9 = arith.addi %8, %7 : tensor<128x128xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<128x128x!tt.ptr<f16>, #src>
    %11 = tt.load %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #src>
    %3 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #dst>
    %12 = triton_gpu.convert_layout %9 : (tensor<128x128xi32, #src>) -> tensor<128x128xi32, #dst>
    %13 = triton_gpu.convert_layout %11 : (tensor<128x128xf16, #src>) -> tensor<128x128xf16, #dst>
    %14 = tt.addptr %3, %12 : tensor<128x128x!tt.ptr<f16>, #dst>
    tt.store %14, %13 : tensor<128x128xf16, #dst>
    return
  }
}    
"""

    x = to_triton(numpy_random(shape, dtype_str=dtype))
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    kernel[(1,1,1)](x.data_ptr(), z.data_ptr())

    assert torch.equal(z, x)

