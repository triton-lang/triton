// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @basic_insert_slice_async_1d(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK: llvm.func @basic_insert_slice_async_1d
    // CHECK-COUNT-64: %{{[0-9]*}} = llvm.extractvalue  %arg0[{{[0-9]*}}] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: %64 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-8:  %{{[0-9]*}} = llvm.insertvalue %{{[0-9]*}}, %{{[0-9]*}}[{{[0-9]*}}] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %72 = amdgpu.view_slice %arg0[0,0] [256, 16] [1,1] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
    tt.return
  }
}
